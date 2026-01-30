"""
DepthCharge: Adaptive Drilling with Verified Ground Truth and Survival Statistics

A domain-agnostic framework for measuring knowledge depth in Large Language Models.

Key innovations:
1. ADAPTIVE DRILLING: Follow-up questions based on model's actual answers
2. ON-DEMAND VERIFICATION: Search for verified facts in real-time as we drill
3. CONSTANT STATISTICAL POWER: Always ask N questions per depth (default 30)
4. SURVIVAL CURVES: Cumulative survival measurement across depths

The drilling process:
1. Start with broad questions about the topic
2. Extract concepts mentioned in model's answer
3. Search for verifiable facts about those concepts
4. Generate follow-up questions from verified facts
5. Repeat, going deeper into whatever the model engages with

The survival guarantee:
- At each depth, we ask exactly N questions (distributed across surviving paths)
- Survival rate = fraction correct at each depth
- Cumulative survival = product of all accuracies up to that depth
"""

import json
import time
import re
import os
import random
import requests
import urllib.parse
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from enum import Enum

import config


class SpecificityTier(Enum):
    """How obscure/specialized the knowledge is."""
    COMMON = 1        # General public (Wikipedia intro)
    TEXTBOOK = 2      # University student (Wikipedia detailed)
    PROFESSIONAL = 3  # Practitioner (clinical guidelines)
    SPECIALIST = 4    # Expert (peer-reviewed literature)
    CUTTING_EDGE = 5  # Researcher (recent papers)


class CognitiveLevel(Enum):
    """What cognitive operation is required."""
    RECALL = 1       # State the fact
    EXPLAIN = 2      # Explain why/how
    APPLY = 3        # Use in scenario
    ANALYZE = 4      # Compare/differentiate


@dataclass
class DrillPath:
    """A single drilling path through the knowledge space."""
    path_id: str
    topic_chain: List[str]  # Topics we've drilled through
    current_topic: str
    previous_answer: Optional[str]
    depth: int
    alive: bool = True


@dataclass
class VerifiedFact:
    """A fact verified from an authoritative source."""
    text: str
    source_url: str
    source_type: str
    topic: str
    subtopic: str
    confidence: float


@dataclass
class ProbeResult:
    """Result of a single probe at a specific depth."""
    depth: int
    level: str  # e.g., "TEXTBOOK/P2"
    path_id: str
    topic_chain: List[str]
    question: str
    verified_answer: str
    model_answer: str
    is_correct: bool
    similarity: float
    source_url: str
    drill_direction: str  # What concept we drilled into


@dataclass
class DepthResult:
    """Aggregate results for a single depth level."""
    depth: int
    level: str
    questions_asked: int
    correct: int
    accuracy_at_depth: float  # correct/asked at this depth
    cumulative_survival: float  # product of all accuracies up to this depth
    probes: List[ProbeResult]


class DepthCharge:
    """
    Adaptive drilling with verified ground truth and survival statistics.

    A domain-agnostic framework that combines:
    - Adaptive follow-up drilling based on model responses
    - On-demand fact verification from authoritative sources
    - Proper survival statistics with constant sample size per depth
    """

    def __init__(self, target_model: str, api_key: str = None,
                 questions_per_depth: int = None,
                 passes_per_tier: int = None,
                 survival_threshold: float = None,
                 max_depth: int = None):
        """
        Initialize DepthCharge.

        Args:
            target_model: Model ID to probe (key from config.MODELS)
            api_key: API key (defaults to config.API_KEY)
            questions_per_depth: Number of questions at each depth (N)
            passes_per_tier: Number of passes per specificity tier (Q)
            survival_threshold: Stop when survival falls below this
            max_depth: Maximum drilling depth
        """
        self.target_model = config.MODELS.get(target_model, target_model)
        self.target_model_name = target_model
        self.extractor_model = config.MODELS.get("extractor", config.MODELS.get(config.DEFAULT_EXTRACTOR_MODEL))
        self.fact_search_model = config.MODELS.get("fact_search", config.MODELS.get(config.DEFAULT_FACT_SEARCH_MODEL))
        self.api_key = api_key or config.API_KEY

        # Use config defaults if not specified
        self.questions_per_depth = questions_per_depth or config.QUESTIONS_PER_DEPTH
        self.passes_per_tier = passes_per_tier or config.PASSES_PER_TIER
        self.survival_threshold = survival_threshold or config.SURVIVAL_THRESHOLD
        self.max_depth = max_depth or config.MAX_DEPTH

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "DepthCharge"
        }

        self.wiki_headers = {
            "User-Agent": "DepthChargeResearch/1.0"
        }

        # Metrics tracking
        self.metrics = {
            "total_calls": 0,
            "total_latency_sec": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "by_model": defaultdict(lambda: {
                "calls": 0,
                "latency_sec": 0.0,
                "input_tokens": 0,
                "output_tokens": 0
            })
        }

        # Cost rates from config
        self.cost_rates = config.COST_RATES

        # Track start time
        self.start_time = None

        # Cache for verified facts (to avoid re-searching)
        self.fact_cache: Dict[str, VerifiedFact] = {}

        # Random generator for reproducibility
        self.rng = random.Random()

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = random.Random(seed)

    def _call_model(self, model_id: str, messages: List[Dict],
                    max_tokens: int = 1000) -> str:
        """Call an LLM via the API."""
        for attempt in range(3):
            try:
                start = time.time()
                payload = {
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": max_tokens
                }
                response = requests.post(
                    f"{config.BASE_URL}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                elapsed = time.time() - start

                self.metrics["total_calls"] += 1
                self.metrics["total_latency_sec"] += elapsed
                self.metrics["by_model"][model_id]["calls"] += 1
                self.metrics["by_model"][model_id]["latency_sec"] += elapsed

                if response.status_code == 200:
                    result = response.json()
                    # Track token usage if available
                    usage = result.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)

                    self.metrics["total_input_tokens"] += input_tokens
                    self.metrics["total_output_tokens"] += output_tokens
                    self.metrics["by_model"][model_id]["input_tokens"] += input_tokens
                    self.metrics["by_model"][model_id]["output_tokens"] += output_tokens

                    return result["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    print(f"      [!] Rate limit. Waiting 5s...")
                    time.sleep(5)
                    continue
                else:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

            except requests.exceptions.Timeout:
                print(f"      [!] Timeout (attempt {attempt + 1}/3)")
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise RuntimeError(f"Request failed: {e}")
                time.sleep(2)

        raise RuntimeError("Max retries exceeded")

    def _get_level(self, depth: int) -> Tuple[SpecificityTier, int]:
        """Map depth to difficulty level.

        Returns:
            Tuple of (SpecificityTier, pass_number) where pass_number is 1-indexed
        """
        # Specificity increases every Q depths (passes_per_tier)
        tier_index = (depth - 1) // self.passes_per_tier
        tiers = [SpecificityTier.COMMON, SpecificityTier.TEXTBOOK,
                 SpecificityTier.PROFESSIONAL, SpecificityTier.SPECIALIST,
                 SpecificityTier.CUTTING_EDGE]
        specificity = tiers[min(tier_index, len(tiers) - 1)]

        # Pass number within the tier (1, 2, 3, ...)
        pass_number = ((depth - 1) % self.passes_per_tier) + 1

        return specificity, pass_number

    def _get_level_string(self, depth: int) -> str:
        """Get human-readable level string."""
        spec, pass_num = self._get_level(depth)
        return f"{spec.name}/P{pass_num}"

    # =========================================================================
    # FACT SEARCH - On-demand verification
    # =========================================================================

    def _search_wikipedia_summary(self, topic: str) -> Optional[VerifiedFact]:
        """Search Wikipedia for basic facts (COMMON tier)."""
        try:
            encoded = urllib.parse.quote(topic.replace(' ', '_'))
            response = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}",
                headers=self.wiki_headers,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                extract = data.get("extract", "")
                if extract and len(extract) > 50:
                    return VerifiedFact(
                        text=extract[:500],
                        source_url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        source_type="wikipedia_summary",
                        topic=topic,
                        subtopic="",
                        confidence=0.90
                    )
        except Exception as e:
            print(f"         [!] Wikipedia search error: {e}")

        return None

    def _search_wikipedia_detailed(self, topic: str, subtopic: str) -> Optional[VerifiedFact]:
        """Search Wikipedia for detailed facts (TEXTBOOK tier)."""
        search_term = f"{topic} {subtopic}" if subtopic else topic

        try:
            params = {
                "action": "query",
                "titles": topic,
                "prop": "extracts",
                "exintro": "false",
                "explaintext": "true",
                "format": "json"
            }
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                headers=self.wiki_headers,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                pages = data.get("query", {}).get("pages", {})

                for page_id, page in pages.items():
                    if page_id != "-1":
                        extract = page.get("extract", "")
                        if extract:
                            # Search for subtopic within the article
                            if subtopic:
                                # Try to find relevant section
                                lower_extract = extract.lower()
                                lower_subtopic = subtopic.lower()
                                idx = lower_extract.find(lower_subtopic)
                                if idx != -1:
                                    # Get surrounding context
                                    start = max(0, idx - 100)
                                    end = min(len(extract), idx + 400)
                                    fact_text = extract[start:end]
                                else:
                                    fact_text = extract[:500]
                            else:
                                fact_text = extract[:500]

                            encoded = urllib.parse.quote(topic.replace(' ', '_'))
                            return VerifiedFact(
                                text=fact_text,
                                source_url=f"https://en.wikipedia.org/wiki/{encoded}",
                                source_type="wikipedia_detail",
                                topic=topic,
                                subtopic=subtopic,
                                confidence=0.85
                            )
        except Exception as e:
            print(f"         [!] Wikipedia detailed search error: {e}")

        return None

    def _search_professional_sources(self, topic: str, subtopic: str,
                                     tier: SpecificityTier) -> Optional[VerifiedFact]:
        """Search for professional/specialist facts using retrieval-augmented system."""
        tier_guidance = {
            SpecificityTier.PROFESSIONAL: "clinical guidelines, medical protocols, professional standards",
            SpecificityTier.SPECIALIST: "peer-reviewed research, molecular mechanisms, specific details",
            SpecificityTier.CUTTING_EDGE: "papers from the past 2 years, recent discoveries"
        }

        guidance = tier_guidance.get(tier, "authoritative sources")
        search_query = f"{topic} {subtopic}" if subtopic else topic

        prompt = f"""Find a verified {tier.name}-level fact about: {search_query}

Requirements:
- Must be from {guidance}
- Must be specific and factual (not general knowledge)
- Must be verifiable

Return format:
FACT: [the specific verified fact]
SOURCE: [source name or type]"""

        try:
            response = self._call_model(self.fact_search_model, [
                {"role": "user", "content": prompt}
            ])

            fact_match = re.search(r'FACT:\s*(.+?)(?:SOURCE:|$)', response, re.DOTALL)
            source_match = re.search(r'SOURCE:\s*(.+?)$', response, re.DOTALL)

            if fact_match:
                return VerifiedFact(
                    text=fact_match.group(1).strip()[:500],
                    source_url="",
                    source_type=f"professional_{tier.name.lower()}",
                    topic=topic,
                    subtopic=subtopic,
                    confidence=0.75 if tier == SpecificityTier.CUTTING_EDGE else 0.80
                )
        except Exception as e:
            print(f"         [!] Professional source search error: {e}")

        return None

    def _search_for_fact(self, topic: str, subtopic: str,
                         tier: SpecificityTier) -> Optional[VerifiedFact]:
        """Search for a verified fact at the appropriate tier."""
        cache_key = f"{topic}:{subtopic}:{tier.name}"
        if cache_key in self.fact_cache:
            return self.fact_cache[cache_key]

        fact = None

        # For COMMON and TEXTBOOK, try Wikipedia first
        if tier in [SpecificityTier.COMMON, SpecificityTier.TEXTBOOK]:
            # First try the subtopic directly
            fact = self._search_wikipedia_summary(subtopic if subtopic else topic)

            # If that fails, try the main topic
            if not fact and subtopic:
                fact = self._search_wikipedia_summary(topic)

            # For TEXTBOOK, also try detailed search
            if not fact and tier == SpecificityTier.TEXTBOOK:
                fact = self._search_wikipedia_detailed(topic, subtopic)
        else:
            # PROFESSIONAL, SPECIALIST, CUTTING_EDGE use professional sources
            fact = self._search_professional_sources(topic, subtopic, tier)

        if fact:
            self.fact_cache[cache_key] = fact

        return fact

    # =========================================================================
    # CONCEPT EXTRACTION - Adaptive drilling
    # =========================================================================

    def _extract_concepts(self, answer: str, current_topic: str) -> List[str]:
        """Extract key concepts from model's answer for drilling."""
        prompt = f"""Extract 5 specific concepts/terms from this answer that could be explored deeper.

TOPIC: {current_topic}
ANSWER: {answer[:1000]}

Return a JSON list of 5 specific concepts mentioned (not generic terms):
["concept1", "concept2", "concept3", "concept4", "concept5"]

Focus on:
- Technical terms
- Specific mechanisms or processes
- Named entities (proteins, drugs, conditions)
- Quantitative claims

Return ONLY the JSON list."""

        try:
            response = self._call_model(self.extractor_model, [
                {"role": "user", "content": prompt}
            ])

            # Extract JSON list
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                concepts = json.loads(match.group(0))
                return [c.strip() for c in concepts if isinstance(c, str) and len(c.strip()) > 2][:5]
        except Exception as e:
            print(f"         [!] Concept extraction error: {e}")

        return []

    def _get_initial_branches(self, topic: str, count: int) -> List[str]:
        """Get initial diverse branches for a topic using domain-appropriate aspects."""
        prompt = f"""For the topic "{topic}", list {count} distinct aspects or subtopics that would be covered in an encyclopedia article. These should be factual aspects that Wikipedia would have information about.

Requirements:
- Each aspect should be specific to this topic's domain
- Aspects should be diverse (cover different facets of the topic)
- Use terminology appropriate to the field
- Each aspect should be a short phrase (2-4 words max)

Return ONLY a JSON list of {count} aspects, like:
["{topic}", "aspect 1", "aspect 2", ...]

The first item should always be the topic itself: "{topic}"."""

        try:
            response = self._call_model(self.extractor_model, [
                {"role": "user", "content": prompt}
            ])

            # Extract JSON list
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                aspects = json.loads(match.group(0))
                # Ensure we have strings and the topic itself is first
                aspects = [a.strip() for a in aspects if isinstance(a, str) and len(a.strip()) > 2]
                if aspects and aspects[0].lower() != topic.lower():
                    aspects = [topic] + aspects
                if len(aspects) >= count:
                    return aspects[:count]
        except Exception as e:
            print(f"         [!] Branch generation error: {e}")

        # Fallback to universal aspects
        universal_aspects = [
            topic,
            f"{topic} history",
            f"{topic} types",
            f"{topic} key concepts",
            f"{topic} notable examples",
            f"{topic} principles",
            f"{topic} development",
            f"{topic} applications",
            f"{topic} significance",
            f"{topic} major figures",
            f"{topic} controversies",
            f"{topic} modern context",
        ]

        return universal_aspects[:count]

    # =========================================================================
    # QUESTION GENERATION & SCORING
    # =========================================================================

    def _generate_question(self, fact: VerifiedFact, previous_answer: Optional[str],
                           tier: SpecificityTier, pass_number: int) -> str:
        """Generate a factual question from a verified fact."""
        tier_descriptions = {
            SpecificityTier.COMMON: "general public knowledge",
            SpecificityTier.TEXTBOOK: "university student level",
            SpecificityTier.PROFESSIONAL: "practitioner level",
            SpecificityTier.SPECIALIST: "expert/specialist level",
            SpecificityTier.CUTTING_EDGE: "cutting-edge research level"
        }

        prompt = f"""Generate a factual question at {tier.name} level ({tier_descriptions[tier]}).

VERIFIED FACT: {fact.text[:400]}
TOPIC: {fact.topic}
SUBTOPIC: {fact.subtopic}

{"PREVIOUS CONTEXT: " + previous_answer[:200] + "..." if previous_answer else ""}

Requirements:
- Question should be answerable using the verified fact
- Ask a direct factual question (What, Which, Name, etc.)
- Difficulty: {tier_descriptions[tier]}
{"- Reference the previous context naturally" if previous_answer else ""}

Return ONLY the question text."""

        try:
            question = self._call_model(self.extractor_model, [
                {"role": "user", "content": prompt}
            ])
            return question.strip().strip('"')
        except Exception:
            # Fallback to simple template
            return f"What is {fact.subtopic or fact.topic}?"

    def _extract_answer_from_fact(self, fact: VerifiedFact, question: str) -> str:
        """Extract the expected answer from the verified fact."""
        prompt = f"""Given this verified fact, what is the answer to the question?

FACT: {fact.text}
QUESTION: {question}

Return ONLY the answer (1-3 sentences, concise)."""

        try:
            return self._call_model(self.extractor_model, [
                {"role": "user", "content": prompt}
            ])
        except Exception:
            return fact.text[:200]

    def _score_answer(self, model_answer: str, correct_answer: str,
                      question: str) -> Tuple[bool, float]:
        """
        Score model's answer using entailment-based evaluation.

        Key principles:
        - Verbose answers that contain the correct information are CORRECT
        - Additional details beyond the expected answer are fine
        - Formatting differences (markdown, bullets) are ignored
        - Focus on factual accuracy, not stylistic similarity
        """
        prompt = f"""You are a factual accuracy checker. Determine if the MODEL ANSWER correctly addresses the question based on the REFERENCE ANSWER.

QUESTION: {question}

REFERENCE ANSWER (verified correct): {correct_answer}

MODEL ANSWER: {model_answer[:600]}

EVALUATION RULES:
1. The model answer is CORRECT if it contains the key factual information from the reference
2. Additional correct details in the model answer are GOOD (don't penalize verbosity)
3. Different wording/phrasing that conveys the same meaning is CORRECT
4. Ignore formatting differences (markdown, bullets, headers)
5. Minor omissions are OK if the core facts are present
6. The model answer is INCORRECT only if it:
   - States factually wrong information
   - Misses critical facts that change the meaning
   - Contradicts the reference answer

Return JSON only: {{"correct": true, "confidence": 0.95, "reason": "contains key facts"}}
or: {{"correct": false, "confidence": 0.9, "reason": "missing X / wrong about Y"}}"""

        try:
            response = self._call_model(self.extractor_model, [
                {"role": "user", "content": prompt}
            ])
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return data.get("correct", False), data.get("confidence", 0.5)
        except Exception as e:
            print(f"         [!] Scoring error: {e}")

        return False, 0.0

    # =========================================================================
    # QUESTION DISTRIBUTION
    # =========================================================================

    def _distribute_questions(self, paths: List[DrillPath],
                              target: int) -> List[Tuple[DrillPath, int]]:
        """Distribute questions across paths to maintain target count."""
        alive_paths = [p for p in paths if p.alive]
        n_paths = len(alive_paths)

        if n_paths == 0:
            return []

        if n_paths >= target:
            # More paths than needed: sample paths, 1 question each
            selected = self.rng.sample(alive_paths, target)
            return [(p, 1) for p in selected]
        else:
            # Fewer paths: multiple questions per path
            base = target // n_paths
            remainder = target % n_paths

            distribution = []
            for i, path in enumerate(alive_paths):
                extra = 1 if i < remainder else 0
                distribution.append((path, base + extra))

            return distribution

    def _get_drill_directions(self, path: DrillPath, count: int,
                              tier: SpecificityTier) -> List[str]:
        """Get multiple drill directions from a path."""
        if path.previous_answer:
            # Extract concepts from previous answer
            concepts = self._extract_concepts(path.previous_answer, path.current_topic)
            if len(concepts) >= count:
                return concepts[:count]

            # Need more directions - generate them
            needed = count - len(concepts)
            prompt = f"""For the topic "{path.current_topic}", suggest {needed} more specific subtopics to explore.

Already exploring: {concepts}

Return a JSON list of {needed} NEW subtopics (different from above):
["subtopic1", "subtopic2", ...]"""

            try:
                response = self._call_model(self.extractor_model, [
                    {"role": "user", "content": prompt}
                ])
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    more = json.loads(match.group(0))
                    concepts.extend([m.strip() for m in more if isinstance(m, str)])
            except Exception:
                pass

            return concepts[:count]
        else:
            # Initial drilling - get diverse aspects
            return self._get_initial_branches(path.current_topic, count)

    # =========================================================================
    # MAIN PROBING LOGIC
    # =========================================================================

    def probe(self, topic: str, seed: int = None) -> Dict:
        """
        Run the full adaptive drilling probe with survival statistics.

        Args:
            topic: The topic to probe
            seed: Random seed for reproducibility (defaults to config.DEFAULT_SEED)

        Returns:
            Dict with survival curve, all probes, and statistics
        """
        seed = seed if seed is not None else config.DEFAULT_SEED
        self.set_seed(seed)
        self.start_time = time.time()

        print("=" * 70)
        print("DepthCharge: Adaptive Drilling with Survival Statistics")
        print("=" * 70)
        print(f"\n[*] Target: {self.target_model}")
        print(f"[*] Topic: {topic}")
        print(f"[*] Questions/Depth (N): {self.questions_per_depth}")
        print(f"[*] Passes/Tier (Q): {self.passes_per_tier}")
        print(f"[*] Survival Threshold: {self.survival_threshold:.0%}")
        print(f"[*] Max Depth: {self.max_depth}")
        print(f"[*] Seed: {seed}")

        # Initialize paths
        initial_branches = self._get_initial_branches(topic, self.questions_per_depth)
        active_paths = [
            DrillPath(
                path_id=f"path_{i}",
                topic_chain=[topic],
                current_topic=branch,
                previous_answer=None,
                depth=0,
                alive=True
            )
            for i, branch in enumerate(initial_branches)
        ]

        results = {
            "topic": topic,
            "target_model": self.target_model,
            "target_model_name": self.target_model_name,
            "questions_per_depth": self.questions_per_depth,
            "passes_per_tier": self.passes_per_tier,
            "survival_threshold": self.survival_threshold,
            "seed": seed,
            "survival_curve": [],
            "all_probes": [],
            "paths_explored": []
        }

        # Track cumulative survival (product of accuracies at each depth)
        cumulative_survival = 1.0

        for depth in range(1, self.max_depth + 1):
            tier, pass_number = self._get_level(depth)
            level_str = self._get_level_string(depth)

            print(f"\n{'─' * 70}")
            print(f"DEPTH {depth}: {level_str}")
            print(f"{'─' * 70}")

            # Distribute questions across paths
            distribution = self._distribute_questions(active_paths, self.questions_per_depth)

            if not distribution:
                print("   [!] No active paths remaining")
                results["stopping_reason"] = "No surviving paths"
                break

            depth_probes = []
            next_paths = []

            for path, num_questions in distribution:
                # Get drill directions for this path
                directions = self._get_drill_directions(path, num_questions, tier)

                for direction in directions:
                    print(f"\n   [{path.path_id}] Drilling: {direction[:40]}...")

                    # Search for verified fact
                    fact = self._search_for_fact(topic, direction, tier)

                    if not fact:
                        print(f"         [!] No verified fact found")
                        probe = ProbeResult(
                            depth=depth,
                            level=level_str,
                            path_id=path.path_id,
                            topic_chain=path.topic_chain + [direction],
                            question="",
                            verified_answer="",
                            model_answer="",
                            is_correct=False,
                            similarity=0.0,
                            source_url="",
                            drill_direction=direction
                        )
                        depth_probes.append(probe)
                        continue

                    # Generate question from verified fact
                    question = self._generate_question(
                        fact, path.previous_answer, tier, pass_number
                    )
                    print(f"         Q: {question[:60]}...")

                    # Extract expected answer
                    correct_answer = self._extract_answer_from_fact(fact, question)

                    # Quiz target model
                    try:
                        model_answer = self._call_model(self.target_model, [
                            {"role": "system", "content": f"You are an expert in {topic}. Answer accurately and concisely."},
                            {"role": "user", "content": question}
                        ])
                    except Exception as e:
                        print(f"         [!] Model error: {e}")
                        model_answer = f"ERROR: {e}"

                    # Score answer
                    is_correct, similarity = self._score_answer(
                        model_answer, correct_answer, question
                    )

                    status = "✓" if is_correct else "✗"
                    print(f"         [{status}] Confidence: {similarity:.2f}")

                    probe = ProbeResult(
                        depth=depth,
                        level=level_str,
                        path_id=path.path_id,
                        topic_chain=path.topic_chain + [direction],
                        question=question,
                        verified_answer=correct_answer[:300],
                        model_answer=model_answer[:300],
                        is_correct=is_correct,
                        similarity=similarity,
                        source_url=fact.source_url,
                        drill_direction=direction
                    )
                    depth_probes.append(probe)

                    # If correct, this branch survives
                    if is_correct:
                        new_path = DrillPath(
                            path_id=f"{path.path_id}_d{depth}_{len(next_paths)}",
                            topic_chain=path.topic_chain + [direction],
                            current_topic=direction,
                            previous_answer=model_answer,
                            depth=depth,
                            alive=True
                        )
                        next_paths.append(new_path)

            # Calculate survival at this depth
            n_asked = len(depth_probes)
            n_correct = sum(1 for p in depth_probes if p.is_correct)
            accuracy_at_depth = n_correct / n_asked if n_asked > 0 else 0

            # Cumulative survival: multiply by accuracy at this depth
            cumulative_survival = cumulative_survival * accuracy_at_depth

            depth_result = DepthResult(
                depth=depth,
                level=level_str,
                questions_asked=n_asked,
                correct=n_correct,
                accuracy_at_depth=accuracy_at_depth,
                cumulative_survival=cumulative_survival,
                probes=depth_probes
            )

            results["survival_curve"].append({
                "depth": depth,
                "level": level_str,
                "questions_asked": n_asked,
                "correct": n_correct,
                "accuracy_at_depth": accuracy_at_depth,
                "cumulative_survival": cumulative_survival
            })

            for probe in depth_probes:
                results["all_probes"].append(asdict(probe))

            print(f"\n   DEPTH ACCURACY: {n_correct}/{n_asked} = {accuracy_at_depth:.0%}")
            print(f"   CUMULATIVE SURVIVAL: {cumulative_survival:.1%}")

            # Check stopping condition based on cumulative survival
            if cumulative_survival < self.survival_threshold:
                results["stopping_reason"] = f"Cumulative survival {cumulative_survival:.0%} below threshold {self.survival_threshold:.0%}"
                print(f"\n   [!] Below threshold - stopping")
                break

            if not next_paths:
                results["stopping_reason"] = "No surviving paths"
                break

            # Update active paths
            active_paths = next_paths

        # Compute statistics
        results["statistics"] = self._compute_statistics(results["survival_curve"])

        # Record explored paths
        all_paths = set()
        for probe in results["all_probes"]:
            all_paths.add(" > ".join(probe["topic_chain"]))
        results["paths_explored"] = list(all_paths)

        # Calculate runtime
        end_time = time.time()
        total_runtime_sec = end_time - self.start_time if self.start_time else self.metrics["total_latency_sec"]

        # Calculate costs
        cost_info = self._calculate_costs()

        # Add metrics
        results["metrics"] = {
            "total_calls": self.metrics["total_calls"],
            "total_latency_sec": round(self.metrics["total_latency_sec"], 2),
            "total_runtime_sec": round(total_runtime_sec, 2),
            "total_input_tokens": self.metrics["total_input_tokens"],
            "total_output_tokens": self.metrics["total_output_tokens"],
            "estimated_cost_usd": cost_info["total_cost_usd"],
            "by_model": {
                k: {
                    "calls": v["calls"],
                    "latency_sec": round(v["latency_sec"], 2),
                    "input_tokens": v["input_tokens"],
                    "output_tokens": v["output_tokens"],
                    "estimated_cost_usd": cost_info["by_model"].get(k, {}).get("total_cost_usd", 0)
                } for k, v in self.metrics["by_model"].items()
            }
        }

        # Print summary
        self._print_summary(results)

        return results

    def _compute_statistics(self, survival_curve: List[Dict]) -> Dict:
        """Compute summary statistics from survival curve."""
        if not survival_curve:
            return {}

        # Expected Valid Depth (EVD): area under cumulative survival curve
        # Sum of cumulative survival probabilities at each depth
        evd = sum(s["cumulative_survival"] for s in survival_curve)

        # Median depth (where cumulative survival drops below 50%)
        median_depth = None
        for s in survival_curve:
            if s["cumulative_survival"] < 0.5:
                median_depth = s["depth"]
                break

        # Max depth with any correct answers
        max_depth_any_correct = 0
        for s in survival_curve:
            if s["correct"] > 0:
                max_depth_any_correct = s["depth"]

        # Final cumulative survival
        final_survival = survival_curve[-1]["cumulative_survival"] if survival_curve else 0

        # Accuracy by tier
        tier_accuracy = {}
        for s in survival_curve:
            tier = s["level"].split("/")[0]
            if tier not in tier_accuracy:
                tier_accuracy[tier] = []
            tier_accuracy[tier].append(s["accuracy_at_depth"])

        tier_summary = {
            tier: round(sum(rates) / len(rates), 3)
            for tier, rates in tier_accuracy.items()
        }

        # Total questions and accuracy
        total_asked = sum(s["questions_asked"] for s in survival_curve)
        total_correct = sum(s["correct"] for s in survival_curve)
        overall_accuracy = total_correct / total_asked if total_asked > 0 else 0

        return {
            "expected_valid_depth": round(evd, 2),
            "median_depth": median_depth,
            "max_depth_with_correct": max_depth_any_correct,
            "final_cumulative_survival": round(final_survival, 3),
            "total_questions": total_asked,
            "total_correct": total_correct,
            "overall_accuracy": round(overall_accuracy, 3),
            "accuracy_by_tier": tier_summary
        }

    def _calculate_costs(self) -> Dict:
        """Calculate API costs based on token usage."""
        total_cost = 0.0
        cost_by_model = {}

        for model_id, usage in self.metrics["by_model"].items():
            rates = self.cost_rates.get(model_id, {"input": 1.0, "output": 1.0})
            input_cost = (usage["input_tokens"] / 1_000_000) * rates["input"]
            output_cost = (usage["output_tokens"] / 1_000_000) * rates["output"]
            model_cost = input_cost + output_cost
            total_cost += model_cost
            cost_by_model[model_id] = {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "input_cost_usd": round(input_cost, 4),
                "output_cost_usd": round(output_cost, 4),
                "total_cost_usd": round(model_cost, 4)
            }

        return {
            "total_cost_usd": round(total_cost, 4),
            "total_input_tokens": self.metrics["total_input_tokens"],
            "total_output_tokens": self.metrics["total_output_tokens"],
            "by_model": cost_by_model
        }

    def _print_summary(self, results: Dict):
        """Print a summary of results."""
        print(f"\n{'=' * 70}")
        print("RESULTS SUMMARY")
        print(f"{'=' * 70}")

        print(f"\nSurvival Curve (Cumulative):")
        print(f"  {'Depth':<6} {'Level':<20} {'Acc.':<6} {'Cumul.':<8} {'Bar':<22} {'Score'}")
        print(f"  {'-'*6} {'-'*20} {'-'*6} {'-'*8} {'-'*22} {'-'*10}")
        for s in results["survival_curve"]:
            bar_len = int(s["cumulative_survival"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  D{s['depth']:<4} {s['level']:<20} {s['accuracy_at_depth']:5.0%}  {s['cumulative_survival']:6.1%}  {bar}  ({s['correct']}/{s['questions_asked']})")

        stats = results.get("statistics", {})
        print(f"\nStatistics:")
        print(f"  Expected Valid Depth (EVD): {stats.get('expected_valid_depth', 'N/A')}")
        print(f"  Median Depth: {stats.get('median_depth', 'N/A')}")
        print(f"  Max Depth with Correct: {stats.get('max_depth_with_correct', 'N/A')}")
        print(f"  Final Cumulative Survival: {stats.get('final_cumulative_survival', 0):.1%}")
        print(f"  Overall Accuracy: {stats.get('overall_accuracy', 0):.1%}")

        print(f"\n  Accuracy by Tier:")
        for tier, rate in stats.get("accuracy_by_tier", {}).items():
            print(f"    {tier:15s}: {rate:.0%}")

        if "stopping_reason" in results:
            print(f"\n  Stopped: {results['stopping_reason']}")

        # Print cost and runtime metrics
        metrics = results.get("metrics", {})
        print(f"\n  Runtime & Cost:")
        print(f"    Total Runtime: {metrics.get('total_runtime_sec', 0):.1f} seconds")
        print(f"    Total API Calls: {metrics.get('total_calls', 0)}")
        print(f"    Total Tokens: {metrics.get('total_input_tokens', 0):,} in / {metrics.get('total_output_tokens', 0):,} out")
        print(f"    Estimated Cost: ${metrics.get('estimated_cost_usd', 0):.4f} USD")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DepthCharge: Adaptive Drilling with Survival Statistics"
    )

    parser.add_argument("--model", type=str, default="model_a",
                        help="Target model key from config.MODELS")
    parser.add_argument("--topic", type=str, default="Medicine",
                        help="Topic to probe")
    parser.add_argument("--questions-per-depth", "-n", type=int, default=None,
                        help=f"Questions per depth level (N). Default: {config.QUESTIONS_PER_DEPTH}")
    parser.add_argument("--passes-per-tier", "-q", type=int, default=None,
                        help=f"Passes per specificity tier (Q). Default: {config.PASSES_PER_TIER}")
    parser.add_argument("--max-depth", type=int, default=None,
                        help=f"Maximum drilling depth. Default: {config.MAX_DEPTH}")
    parser.add_argument("--survival-threshold", type=float, default=None,
                        help=f"Stop when survival falls below this. Default: {config.SURVIVAL_THRESHOLD}")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Random seed. Default: {config.DEFAULT_SEED}")
    parser.add_argument("--output", type=str, default="results/probe_output.json",
                        help="Output file")

    args = parser.parse_args()

    prober = DepthCharge(
        target_model=args.model,
        questions_per_depth=args.questions_per_depth,
        passes_per_tier=args.passes_per_tier,
        max_depth=args.max_depth,
        survival_threshold=args.survival_threshold
    )

    results = prober.probe(args.topic, seed=args.seed)

    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[*] Results saved to {args.output}")
