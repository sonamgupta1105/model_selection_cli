#!/usr/bin/env python3
import os
import time
import json
import asyncio
import aiohttp
from typing import List, Dict, Any
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from tabulate import tabulate
from dotenv import load_dotenv
import statistics

# Load environment variables
load_dotenv()

# Constants
API_BASE_URL = "https://api.hyperbolic.xyz/v1"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzb25hbS5ndXB0YTExMDVAZ21haWwuY29tIiwiaWF0IjoxNzQ2NTA4MDU1fQ.Am0PtuOoUfI0s8t02F8M231N8idlXht47OXuYHTszv8"

# Test prompts for accuracy evaluation
TEST_PROMPTS = [
    # MMLU-style questions
    "Q: What is the capital of France?\nA: The capital of France is",
    "Q: What is the speed of light in meters per second?\nA: The speed of light is",
    "Q: Who wrote 'Romeo and Juliet'?\nA: 'Romeo and Juliet' was written by",
    # HumanEval-style coding questions
    "Write a Python function to calculate the factorial of a number.",
    "Write a Python function to check if a string is a palindrome.",
    # General knowledge
    "Explain how photosynthesis works in simple terms.",
    "What are the main causes of climate change?",
    # Consistency check prompts
    "Count from 1 to 5.",
    "List the days of the week.",
    "Name the first three planets from the sun."
]

# Expected answers for accuracy checking
EXPECTED_ANSWERS = {
    "capital": "Paris",
    "speed_of_light": "299792458",
    "romeo": "Shakespeare",
    "factorial": ["def factorial", "return", "recursion" or "for" or "while"],
    "palindrome": ["def", "return", "[::-1]" or "reverse"],
    "count": ["1", "2", "3", "4", "5"],
    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "planets": ["Mercury", "Venus", "Earth"]
}

# Model pricing in dollars per 1M tokens
MODEL_PRICING = {
    "llama-3.1-8b": {"price": 0.1},
    "llama-3.3-70b": {"price": 0.4},
    "llama-3.1-70b": {"price": 0.4},
    "llama-3-70b": {"price": 0.4},
    "hermes-3-70b": {"price": 0.4},
    "llama-3.1-405b": {"price": 4.0},
    "deepseek-v2.5": {"price": 2.0},
    "qwen-2.5-72b": {"price": 0.4},
    "llama-3.2-3b": {"price": 0.1},
    "qwen-2.5-coder-32b": {"price": 0.2},
    "qwen-qwq-preview-32b": {"price": 0.2},
    "deepseek-v3": {"price": 0.25},
    "deepseek-r1-zero": {"price": 2.0},
    "deepseek-r1": {"price": 2.0},
    "llama-3.1-405b-base": {"price": 4.0},
    "llama-3.2-90b-vision-base": {"price": 2.0}
}

console = Console()

def get_model_price(model_id: str) -> float:
    """Get the price per 1M tokens for a model."""
    # Convert model ID to lowercase and remove special characters for matching
    model_key = model_id.lower().replace("/", "-").replace("_", "-")
    
    # Try to find a matching model
    for pricing_key, pricing_data in MODEL_PRICING.items():
        if pricing_key in model_key:
            return pricing_data["price"]
    
    # Default price if not found
    return 0.4  # Default to standard 70B model price

def calculate_response_accuracy(response: str, expected_keywords: List[str]) -> float:
    """Calculate accuracy based on presence of expected keywords."""
    if not response or not expected_keywords:
        return 0.0
    response_lower = response.lower()
    matches = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
    return matches / len(expected_keywords)

async def evaluate_accuracy(session: aiohttp.ClientSession, model_id: str) -> Dict[str, float]:
    """Evaluate model accuracy using test prompts."""
    accuracies = []
    consistency_scores = []
    
    try:
        # Run each prompt 3 times for consistency check
        for prompt in TEST_PROMPTS:
            responses = []
            for _ in range(3):
                try:
                    result = await measure_model_performance(session, model_id, prompt)
                    if result and result.get('response'):
                        responses.append(result['response'])
                except Exception as e:
                    console.print(f"[yellow]Warning: Error in prompt evaluation: {str(e)}[/yellow]")
                    continue
            
            if not responses:
                continue
                
            # Calculate consistency score
            if len(responses) > 1:
                if len(set(responses)) == 1:
                    consistency_scores.append(1.0)  # Perfect consistency
                else:
                    # Calculate similarity between responses
                    similarity_scores = []
                    for i in range(len(responses)):
                        for j in range(i + 1, len(responses)):
                            words1 = set(responses[i].split())
                            words2 = set(responses[j].split())
                            if words1 or words2:  # Avoid division by zero
                                common_words = words1 & words2
                                total_words = words1 | words2
                                similarity_scores.append(len(common_words) / len(total_words))
                    if similarity_scores:
                        consistency_scores.append(statistics.mean(similarity_scores))
                    else:
                        consistency_scores.append(0.0)
            
            # Calculate accuracy for known-answer questions
            if responses:
                if "capital" in prompt.lower():
                    accuracies.append(calculate_response_accuracy(responses[0], [EXPECTED_ANSWERS["capital"]]))
                elif "speed of light" in prompt.lower():
                    accuracies.append(calculate_response_accuracy(responses[0], [EXPECTED_ANSWERS["speed_of_light"]]))
                elif "Romeo and Juliet" in prompt:
                    accuracies.append(calculate_response_accuracy(responses[0], [EXPECTED_ANSWERS["romeo"]]))
                elif "factorial" in prompt:
                    accuracies.append(calculate_response_accuracy(responses[0], EXPECTED_ANSWERS["factorial"]))
                elif "palindrome" in prompt:
                    accuracies.append(calculate_response_accuracy(responses[0], EXPECTED_ANSWERS["palindrome"]))
                elif "Count from 1 to 5" in prompt:
                    accuracies.append(calculate_response_accuracy(responses[0], EXPECTED_ANSWERS["count"]))
                elif "days of the week" in prompt:
                    accuracies.append(calculate_response_accuracy(responses[0], EXPECTED_ANSWERS["days"]))
                elif "planets" in prompt:
                    accuracies.append(calculate_response_accuracy(responses[0], EXPECTED_ANSWERS["planets"]))
        
        # Calculate final scores with error handling
        mmlu_score = statistics.mean(accuracies[:3]) * 100 if accuracies[:3] else 0.0
        humaneval_score = statistics.mean(accuracies[3:5]) * 100 if accuracies[3:5] else 0.0
        consistency_score = statistics.mean(consistency_scores) * 100 if consistency_scores else 0.0
        
        return {
            "mmlu": mmlu_score,
            "humaneval": humaneval_score,
            "consistency": consistency_score
        }
    except Exception as e:
        console.print(f"[yellow]Warning: Error in accuracy evaluation: {str(e)}[/yellow]")
        return {
            "mmlu": 0.0,
            "humaneval": 0.0,
            "consistency": 0.0
        }

async def measure_model_performance(session: aiohttp.ClientSession, model_id: str, prompt: str) -> Dict[str, Any]:
    """Measure performance metrics for a given model."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 0.0,
        "stream": True
    }
    
    start_time = time.time()
    first_token_time = None
    total_tokens = 0
    response_text = ""
    
    async with session.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=payload) as response:
        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"API Error ({response.status}): {error_text}")
            
        async for line in response.content:
            if line:
                try:
                    line_text = line.decode('utf-8').strip()
                    if line_text.startswith('data: '):
                        line_text = line_text[6:]
                    if line_text == '[DONE]':
                        continue
                    data = json.loads(line_text)
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    if 'choices' in data and data['choices'][0].get('delta', {}).get('content'):
                        content = data['choices'][0]['delta']['content']
                        response_text += content
                        total_tokens += 1
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    console.print(f"[yellow]Warning: Error processing line: {str(e)}[/yellow]")
                    continue
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate costs
    price_per_1m = get_model_price(model_id)
    price_per_1k = price_per_1m / 1000  # Convert to price per 1K tokens
    
    return {
        "time_to_first_token": first_token_time * 1000 if first_token_time else 0,  # Convert to ms
        "total_latency": total_time * 1000,  # Convert to ms
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
        "response": response_text,
        "total_tokens": total_tokens,
        "price_per_1k": price_per_1k
    }

def format_metric(value1: float, value2: float, format_str: str = "{:.1f}") -> tuple[str, str]:
    """Format metric values and determine the better one."""
    str1 = format_str.format(value1)
    str2 = format_str.format(value2)
    return str1, str2

@click.command()
@click.argument('model1')
@click.argument('model2')
@click.option('--prompt', default="Explain quantum computing in simple terms.", help="Test prompt to use for comparison")
def cli(model1: str, model2: str, prompt: str):
    """Compare two LLM models hosted on Hyperbolic API."""
    asyncio.run(compare_models(model1, model2, prompt))

async def compare_models(model1: str, model2: str, prompt: str):
    """Compare two LLM models hosted on Hyperbolic API."""
    if not API_KEY:
        console.print("[red]Error: API key not found[/red]")
        return

    async with aiohttp.ClientSession() as session:
        try:
            # Measure performance
            console.print(f"[cyan]Testing {model1}...[/cyan]")
            results1 = await measure_model_performance(session, model1, prompt)
            accuracy1 = await evaluate_accuracy(session, model1)
            
            console.print(f"[cyan]Testing {model2}...[/cyan]")
            results2 = await measure_model_performance(session, model2, prompt)
            accuracy2 = await evaluate_accuracy(session, model2)
            
            # # Display model outputs for the main prompt
            # console.print("\n[bold cyan]Model Outputs for Main Prompt:[/bold cyan]")
            # console.print(Panel(results1['response'], title=f"{model1} Response", style="green"))
            # console.print(Panel(results2['response'], title=f"{model2} Response", style="blue"))
            # console.print()
            
            # Format metrics
            ttft1, ttft2 = format_metric(results1['time_to_first_token'], results2['time_to_first_token'])
            lat1, lat2 = format_metric(results1['total_latency'] / 1000, results2['total_latency'] / 1000, "{:.1f}")
            tps1, tps2 = format_metric(results1['tokens_per_second'], results2['tokens_per_second'])
            
            # Display comparison in the requested format
            console.print("\nSpeed Metrics:")
            console.print(f"Time to first token: {ttft1}ms vs {ttft2}ms")
            console.print(f"Total latency: {lat1}s vs {lat2}s")
            console.print(f"Tokens/sec: {tps1} vs {tps2}")
            
            console.print("\nAccuracy Metrics:")
            console.print(f"MMLU score: {accuracy1['mmlu']:.1f}% vs {accuracy2['mmlu']:.1f}%")
            console.print(f"HumanEval: {accuracy1['humaneval']:.1f}% vs {accuracy2['humaneval']:.1f}%")
            console.print(f"Consistency: {accuracy1['consistency']:.1f}% vs {accuracy2['consistency']:.1f}%")
            
            console.print("\nCost Analysis:")
            price1 = results1['price_per_1k']
            price2 = results2['price_per_1k']
            console.print(f"Cost per 1K tokens: ${price1:.4f} vs ${price2:.4f}")
            
            # Calculate cost for a standard workload (1M tokens)
            std_workload = 1_000_000
            workload_cost1 = (std_workload / 1000) * price1
            workload_cost2 = (std_workload / 1000) * price2
            console.print(f"Cost for 1M tokens: ${workload_cost1:.2f} vs ${workload_cost2:.2f}")
            
            # Calculate cost-performance ratio (tokens/sec per dollar)
            try:
                perf_ratio1 = results1['tokens_per_second'] / price1 if price1 > 0 else results1['tokens_per_second']
                perf_ratio2 = results2['tokens_per_second'] / price2 if price2 > 0 else results2['tokens_per_second']
                ratio1, ratio2 = format_metric(perf_ratio1, perf_ratio2, "{:.1f}")
                console.print(f"Cost-performance ratio: {ratio1}x vs {ratio2}x")
            except Exception as e:
                console.print("[yellow]Warning: Could not calculate cost-performance ratio[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")

if __name__ == "__main__":
    cli()
