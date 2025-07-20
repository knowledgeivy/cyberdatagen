# src/cyberdata/scripts/batch1_generate_synthetic.py

import pandas as pd
import numpy as np
import json
import time
import concurrent.futures
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv

# Determine PROJECT_ROOT
current_file = Path(__file__).resolve()
if 'src/cyberdata/scripts' in str(current_file):
    PROJECT_ROOT = current_file.parents[3]
else:
    PROJECT_ROOT = Path.cwd()

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# Load environment
load_dotenv()
client = OpenAI()

# Paths
BATCH1_DIR = PROJECT_ROOT / 'data/batch1'
SEED_SAMPLES_DIR = BATCH1_DIR / 'seed_samples'
SYNTHETIC_DIR = BATCH1_DIR / 'synthetic'

# Create directories
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration
MODEL_NAME = "gpt-4o-mini"
MAX_WORKERS = 10

# Rewrite prompts
PROMPTS = {
    'rewrite': {
        'system': "You are an expert in email phishing simulation. Rewrite phishing emails to maintain the malicious intent while changing language, structure, and specific details.",
        'user': "Rewrite this phishing email. Keep the malicious intent and core attack vector, but change the wording, structure, and specific details. Return only the rewritten email in the same format.\n\nOriginal email:\nSubject: {subject}\nBody: {body}\n\nRewritten email:"
    },
    'rewrite_strong': {
        'system': "You are an expert in advanced phishing email generation. Create sophisticated variants that are harder to detect while maintaining the original attack intent.",
        'user': "Create a sophisticated variant of this phishing email. Make it more convincing and harder to detect while keeping the same attack goal. Use professional language and realistic scenarios.\n\nOriginal email:\nSubject: {subject}\nBody: {body}\n\nSophisticated variant:"
    },
    'rewrite_weak': {
        'system': "You are generating basic phishing email variants. Create simpler versions that are easier to detect while maintaining the core malicious intent.",
        'user': "Create a simpler, more obvious variant of this phishing email. Make it easier to detect with clear warning signs while keeping the same basic attack intent.\n\nOriginal email:\nSubject: {subject}\nBody: {body}\n\nSimpler variant:"
    }
}

def call_llm(system_prompt, user_prompt, max_retries=3):
    """Call OpenAI API with retry logic"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)

def parse_email_response(response_text, original_subject, original_body):
    """Parse LLM response to extract subject and body"""
    try:
        lines = response_text.strip().split('\n')
        subject = ""
        body = ""
        
        for i, line in enumerate(lines):
            if line.lower().startswith('subject:'):
                subject = line[8:].strip()
                body_lines = []
                for j in range(i + 1, len(lines)):
                    if lines[j].lower().startswith('body:'):
                        body_lines = lines[j + 1:]
                        break
                    elif not lines[j].strip():
                        continue
                    else:
                        body_lines = lines[i + 1:]
                        break
                body = '\n'.join(body_lines).strip()
                break
        
        # If parsing failed, try alternative parsing
        if not subject:
            if 'subject:' in response_text.lower():
                parts = response_text.split('Subject:', 1)
                if len(parts) > 1:
                    remaining = parts[1]
                    if 'Body:' in remaining:
                        subject_part, body_part = remaining.split('Body:', 1)
                        subject = subject_part.strip()
                        body = body_part.strip()
                    else:
                        lines = remaining.strip().split('\n')
                        subject = lines[0].strip() if lines else ""
                        body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
        
        # Fallback: use original if parsing completely failed
        if not subject:
            subject = original_subject
        if not body:
            body = original_body
            
        return subject, body
    except Exception:
        return original_subject, original_body

def generate_synthetic_sample(seed_sample, variant_type, sample_idx):
    """Generate single synthetic sample"""
    try:
        original_subject = str(seed_sample['subject'])
        original_body = str(seed_sample['body'])
        original_id = seed_sample['data_id']
        
        prompt_config = PROMPTS[variant_type]
        system_prompt = prompt_config['system']
        user_prompt = prompt_config['user'].format(
            subject=original_subject,
            body=original_body
        )
        
        response = call_llm(system_prompt, user_prompt)
        subject, body = parse_email_response(response, original_subject, original_body)
        
        # Generate synthetic ID
        seed_idx = original_id.split('_')[-1]
        synthetic_id = f"CEAS08_SYN1_{variant_type.upper()[:2]}_{seed_idx}_{sample_idx:02d}"
        
        return {
            'data_id': synthetic_id,
            'subject': subject,
            'body': body,
            'label': 1,
            'original_id': original_id,
            'variant_type': variant_type,
            'generation_success': True
        }
    except Exception as e:
        return {
            'data_id': f"FAILED_{variant_type}_{sample_idx}",
            'subject': seed_sample['subject'],
            'body': seed_sample['body'],
            'label': 1,
            'original_id': seed_sample['data_id'],
            'variant_type': variant_type,
            'generation_success': False,
            'error': str(e)
        }

def generate_variant_batch(seed_samples, variant_type):
    """Generate one variant type for all seeds"""
    print(f"Generating {variant_type} variants...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, (_, seed_sample) in enumerate(seed_samples.iterrows()):
            future = executor.submit(generate_synthetic_sample, seed_sample, variant_type, idx)
            futures.append(future)
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            if (i + 1) % 50 == 0:
                print(f"  Completed {i + 1}/{len(futures)} samples")
    
    df = pd.DataFrame(results)
    
    # Save results
    output_file = SYNTHETIC_DIR / f'malicious_{variant_type}_1k.csv'
    df.to_csv(output_file, index=False)
    
    success_count = df['generation_success'].sum()
    print(f"  Saved {len(df)} samples ({success_count} successful) to {output_file}")
    
    return df

def main():
    print("=== BATCH 1: GENERATE SYNTHETIC DATA ===")
    
    # Load seed samples
    seed_file = SEED_SAMPLES_DIR / 'malicious_seeds_1k.csv'
    if not seed_file.exists():
        print(f"Error: Seed file not found: {seed_file}")
        print("Please run batch1_prepare_base_data.py first")
        return
    
    seed_samples = pd.read_csv(seed_file)
    print(f"Loaded {len(seed_samples)} seed samples")
    
    # Generate variants
    variant_types = ['rewrite', 'rewrite_strong', 'rewrite_weak']
    results = {}
    
    for variant_type in variant_types:
        results[variant_type] = generate_variant_batch(seed_samples, variant_type)
    
    # Save generation metadata
    metadata = {
        'experiment': 'batch1_synthetic_generation',
        'model': MODEL_NAME,
        'seed_count': len(seed_samples),
        'variant_types': variant_types,
        'max_workers': MAX_WORKERS,
        'generation_timestamp': pd.Timestamp.now().isoformat(),
        'results_summary': {
            variant: {
                'total_generated': len(results[variant]),
                'successful_generations': results[variant]['generation_success'].sum(),
                'success_rate': results[variant]['generation_success'].mean()
            }
            for variant in variant_types
        }
    }
    
    with open(SYNTHETIC_DIR / 'generation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGeneration complete:")
    for variant in variant_types:
        success_rate = results[variant]['generation_success'].mean() * 100
        print(f"- {variant}: {success_rate:.1f}% success rate")
    print(f"- Output directory: {SYNTHETIC_DIR}")

if __name__ == "__main__":
    main()