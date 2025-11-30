# coding: UTF-8
"""
Technical Debt (TD) Detection using ChatGPT API (Few-Shot Enhanced with English Prompts)
"""

import pandas as pd
import openai
import time
import json
import os
import argparse
from typing import List, Dict, Any
import logging
import re

# 设置日志 (日志保留中文，方便查看运行状态)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatGPTTDDetector:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the ChatGPT TD Detector
        """
        self.client = openai.OpenAI(
            api_key=api_key, 
            base_url="https://api.chatanywhere.tech/v1"
        )
        self.model = model
        
        # few-shot example cache
        self.few_shot_examples: List[Dict[str, str]] = []
        
        # ----------------------------------------------------------------------
        # SYSTEM PROMPT (Converted to English)
        # ----------------------------------------------------------------------
        self.system_prompt = """
        You are a professional software quality analysis expert. Please analyze the given Java method code and determine if Technical Debt (TD) exists.

        Common characteristics of Technical Debt include:
        1. Code Duplication
        2. Long Method/Class
        3. Complex Conditional
        4. Lack of Documentation
        5. Hard-coded Values
        6. Poor Exception Handling
        7. Multiple Responsibilities
        8. Poor Naming
        9. Magic Numbers
        10. Obsolete Collection Usage (e.g., Vector, Stack)
        11. Returning Null (instead of Optional or empty collection)
        12. Deep Nesting

        Please strictly return the result in the following JSON format without Markdown formatting:
        {
            "has_td": true,
            "td_types": ["Type1", "Type2"],
            "confidence": 0.95,
            "explanation": "Brief explanation of the findings.",
            "suggestions": ["Suggestion 1", "Suggestion 2"]
        }
        """

    def add_few_shot_example(self, code: str, label_json: Dict[str, Any]):
        """
        Add a few-shot example.
        """
        # Format input and output in English
        user_content = f"Analyze the following Java code:\n```java\n{code}\n```"
        assistant_content = json.dumps(label_json, ensure_ascii=False)
        
        self.few_shot_examples.append({
            "role": "user",
            "content": user_content
        })
        self.few_shot_examples.append({
            "role": "assistant", 
            "content": assistant_content
        })

    def detect_td(self, method_code: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Detect Technical Debt for a single method
        """
        
        # 1. Build message chain: System -> Few-Shot Examples -> Current Target
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add history examples
        if self.few_shot_examples:
            messages.extend(self.few_shot_examples)
            
        # Add current code to analyze
        messages.append({
            "role": "user", 
            "content": f"Analyze the following Java code:\n```java\n{method_code}\n```"
        })
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1, # Low temperature for consistent JSON output
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content.strip()
                
                # Try to parse JSON
                try:
                    # Clean potential markdown code blocks
                    clean_text = re.sub(r"```[a-zA-Z]*", "", content)
                    clean_text = clean_text.replace("```", "").strip()
                    result = json.loads(clean_text)
                    
                    if isinstance(result, dict):
                        return result
                    else:
                        raise ValueError("JSON result is not a dictionary")
                        
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"JSON parse failed, trying text extraction: {content}")
                    return self._extract_info_from_text(content)
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "has_td": False,
                        "td_types": [],
                        "confidence": 0.0,
                        "explanation": f"Detection failed: {str(e)}",
                        "suggestions": []
                    }
        
        return {
            "has_td": False,
            "td_types": [],
            "confidence": 0.0,
            "explanation": "Detection failed: Max retries exceeded",
            "suggestions": []
        }
    
    def _extract_info_from_text(self, text: str) -> Dict[str, Any]:
        """
        Fallback extraction if JSON parsing fails
        """
        text_lower = text.lower()
        # Updated to check for English keywords
        has_td = "true" in text_lower or "yes" in text_lower or "exists" in text_lower
        confidence = 0.5
        
        confidence_match = re.search(r'confidence["\']?\s*[:=]\s*([0-9.]+)', text_lower)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
            except:
                pass
        
        return {
            "has_td": has_td,
            "td_types": ["ParseError"],
            "confidence": confidence,
            "explanation": text[:500],
            "suggestions": []
        }

    def batch_detect(self, csv_file: str, output_file: str = None, 
                    start_idx: int = 0, end_idx: int = None, 
                    delay: float = 1.0) -> pd.DataFrame:
        """
        Batch detection from CSV
        """
        logger.info(f"Reading CSV file: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return pd.DataFrame()
        
        if end_idx is None:
            end_idx = len(df)
        
        df_subset = df.iloc[start_idx:end_idx].copy()
        logger.info(f"Processing range: {start_idx} - {end_idx}, Total: {len(df_subset)}")
        
        results = []
        
        for idx, row in df_subset.iterrows():
            logger.info(f"Processing {idx + 1}/{len(df)} (Index: {idx})")
            
            method_code = row.get('method_code', '')
            project = row.get('project', 'unknown')
            file_name = row.get('file_name', 'unknown')
            original_label = row.get('label', False)
            
            if not isinstance(method_code, str) or not method_code.strip():
                logger.warning(f"Skipping empty code at index: {idx}")
                continue

            # Detect
            detection_result = self.detect_td(method_code)
            
            # Record result
            result = {
                'original_index': idx,
                'project': project,
                'file_name': file_name,
                'method_code': method_code,
                'original_label': original_label,
                'chatgpt_has_td': detection_result.get('has_td', False),
                'td_types': json.dumps(detection_result.get('td_types', []), ensure_ascii=False),
                'confidence': detection_result.get('confidence', 0.0),
                'explanation': detection_result.get('explanation', ''),
                'suggestions': json.dumps(detection_result.get('suggestions', []), ensure_ascii=False)
            }
            
            results.append(result)
            
            if delay > 0:
                time.sleep(delay)
        
        result_df = pd.DataFrame(results)
        
        if output_file:
            result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"Results saved to: {output_file}")
        
        return result_df

def main():
    parser = argparse.ArgumentParser(description='ChatGPT Technical Debt Detector (Few-Shot / English Prompts)')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API Key')
    parser.add_argument('--input_file', type=str, default='dataset/td-dataset.csv', help='Input CSV path')
    parser.add_argument('--output_file', type=str, default='chatgpt_td_results.csv', help='Output CSV path')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model name')
    parser.add_argument('--test_mode', action='store_true', help='Test mode (first 10 rows)')
    
    args = parser.parse_args()
    
    detector = ChatGPTTDDetector(api_key=args.api_key, model=args.model)
    
    # ---------------------------------------------------------
    # Add Few-Shot Examples (Converted to English)
    # ---------------------------------------------------------
    
    # Example 1: The specific TD code you provided
    ex1_code = "public RuntimeConfigurable2 currentWrapper() { if (wStack.size() < 1) return null; return (RuntimeConfigurable2) wStack.elementAt(wStack.size() - 1); }"
    
    ex1_label = {
        "has_td": True,
        "td_types": ["Returning Null", "Obsolete Collection Usage", "Magic Numbers"],
        "confidence": 1.0,
        "explanation": "1. Returns null directly when stack is empty, which may cause NullPointerException. 2. Uses legacy collection methods like elementAt/Stack. 3. Uses a hard-coded number 1.",
        "suggestions": [
            "Use Optional<T> instead of returning null.",
            "Use ArrayList or Deque instead of Stack.",
            "Define the number 1 as a named constant."
        ]
    }
    detector.add_few_shot_example(ex1_code, ex1_label)

    # Example 2: Clean Code (Non-TD)
    ex2_code = """
    public int calculateSum(int a, int b) {
        return a + b;
    }
    """
    ex2_label = {
        "has_td": False,
        "td_types": [],
        "confidence": 1.0,
        "explanation": "The code is concise, the logic is clear, naming is standard, and there is no obvious technical debt.",
        "suggestions": []
    }
    detector.add_few_shot_example(ex2_code, ex2_label)
    
    logger.info(f"Loaded {len(detector.few_shot_examples)//2} Few-Shot examples")

    # ---------------------------------------------------------
    
    try:
        # Create dummy data for test mode if file missing
        if not os.path.exists(args.input_file) and args.test_mode:
            logger.warning("Input file not found, creating test data...")
            test_df = pd.DataFrame({
                'project': ['test_proj'],
                'file_name': ['Test.java'],
                'method_code': ['public void test() { int i=0; if(i==1){ return; } }'],
                'label': [True]
            })
            test_df.to_csv(args.input_file, index=False)

        results = detector.batch_detect(
            csv_file=args.input_file,
            output_file=args.output_file,
            end_idx=5 if args.test_mode else None,
            delay=1.0
        )
        
        if not results.empty and 'original_label' in results.columns:
            try:
                res_bool = results['chatgpt_has_td'].astype(bool)
                orig_bool = results['original_label'].astype(bool)
                correct = (res_bool == orig_bool).sum()
                total = len(results)
                accuracy = correct / total if total > 0 else 0
                logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
            except Exception as e:
                logger.warning(f"Could not calculate accuracy: {e}")
        
        logger.info("Detection Completed!")
        
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")

if __name__ == '__main__':
    main()