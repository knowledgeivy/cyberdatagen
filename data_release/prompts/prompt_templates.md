# Prompt Templates

This document contains the complete prompt templates used to generate synthetic spam samples.

## Prompt Strategies

Three systematically varied prompt strategies were used to explore how generation instructions influence synthetic data quality:

### 1. Original Strategy

**Purpose**: Faithfully reproduce spam characteristics while introducing natural lexical variation.

**System Prompt**:
```
You are an expert at rewriting phishing email content while preserving the original malicious intent and meaning.

Your task is to rewrite phishing emails to create variations that:
- Keep the exact same malicious meaning and intent
- Maintain the same phishing techniques and social engineering tactics
- Use different wording and phrasing
- Preserve the original structure and key elements

Always return valid JSON with "rewritten_subject" and "rewritten_body" fields.
```

**User Prompt Template**:
```
Rewrite this phishing email but keep the content very similar and do not change the meaning:

**Original Subject:** {subject}

**Original Body:** {body}

Return only valid JSON:
{
  "rewritten_subject": "rewritten subject here",
  "rewritten_body": "rewritten body here"
}
```

---

### 2. Strong Strategy

**Purpose**: Explicitly amplify spam indicators by emphasizing representative phishing characteristics.

**System Prompt**:
```
You are an expert at rewriting phishing email content while preserving the original malicious intent and meaning.

Your task is to rewrite phishing emails to create strong, representative variations that:
- Keep the exact same malicious meaning and intent
- Maintain the same phishing techniques and social engineering tactics
- Use different wording and phrasing
- Preserve the original structure and key elements
- Make the rewritten data strong to reflect representative phishing email characteristics

Always return valid JSON with "rewritten_subject" and "rewritten_body" fields.
```

**User Prompt Template**:
```
Rewrite this phishing email but keep the content very similar and do not change the meaning. Make the rewritten data strong to reflect the representative phishing email:

**Original Subject:** {subject}

**Original Body:** {body}

Return only valid JSON:
{
  "rewritten_subject": "rewritten subject here",
  "rewritten_body": "rewritten body here"
}
```

---

### 3. Weak Strategy

**Purpose**: Produce subtle spam with reduced explicit indicators.

**System Prompt**:
```
You are an expert at rewriting phishing email content while preserving the original malicious intent and meaning.

Your task is to rewrite phishing emails to create subtle, less obvious variations that:
- Keep the exact same malicious meaning and intent
- Maintain the same phishing techniques and social engineering tactics
- Use different wording and phrasing
- Preserve the original structure and key elements
- Make the rewritten data weaker and less representative of obvious phishing email characteristics

Always return valid JSON with "rewritten_subject" and "rewritten_body" fields.
```

**User Prompt Template**:
```
Rewrite this phishing email but keep the content very similar and do not change the meaning. Make the rewritten data weaker and less obvious to reflect subtle phishing attempts:

**Original Subject:** {subject}

**Original Body:** {body}

Return only valid JSON:
{
  "rewritten_subject": "rewritten subject here",
  "rewritten_body": "rewritten body here"
}
```

## Model Configurations

### GPT-4.1-mini
```yaml
model: gpt-4.1-mini
max_tokens: 1000
temperature: 0.7
top_p: 1.0
frequency_penalty: 0.0
presence_penalty: 0.0
```

### Claude-3.5-Haiku
```yaml
model: claude-3-5-haiku-20241022
max_tokens: 1000
temperature: 0.7
top_p: 1.0
```

## Quality Control

Generated synthetic samples underwent quality control checks:
- **Length ratio**: 0.3 - 3.0 (30% to 300% of original)
- **Similarity threshold**: 0.1 - 0.95 (prevent exact copies or complete divergence)
- **Language quality score**: ≥ 0.8

Invalid samples were regenerated until passing all checks.
