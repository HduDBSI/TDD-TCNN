# coding: UTF-8
"""
使用ChatGPT API进行方法级别的技术债务(TD)检测
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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatGPTTDDetector:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        初始化ChatGPT TD检测器
        
        Args:
            api_key: OpenAI API密钥
            model: 使用的模型名称
        """
        self.client = openai.OpenAI(
            api_key="sk-xxx",
            base_url="https://api.chatanywhere.tech/v1"
        )
        
    def create_td_detection_prompt(self, method_code: str) -> str:
        """
        创建TD检测的提示词
        
        Args:
            method_code: 要分析的方法代码
            
        Returns:
            格式化的提示词
        """
        prompt = f"""
        你是一个专业的软件质量分析专家。请分析以下Java方法代码，判断是否存在技术债务(Technical Debt)。

        技术债务的常见特征包括：
        1. 代码重复(Code Duplication)
        2. 过长的方法或类(Long Method/Class)
        3. 复杂的条件语句(Complex Conditional)
        4. 缺乏注释或文档(Lack of Documentation)
        5. 硬编码值(Hard-coded Values)
        6. 异常处理不当(Poor Exception Handling)
        7. 方法职责过多(Multiple Responsibilities)
        8. 命名不规范(Poor Naming)
        9. 魔法数字(Magic Numbers)
        10. 深层嵌套(Deep Nesting)

        请分析以下代码：

        ```java
        {method_code}
        ```

        请按照以下JSON格式返回结果：
        {{
            "has_td": true/false,
            "td_types": ["类型1", "类型2"],
            "confidence": 0.85,
            "explanation": "详细的解释说明",
            "suggestions": ["改进建议1", "改进建议2"]
        }}

        其中：
        - has_td: 是否存在技术债务
        - td_types: 检测到的技术债务类型列表
        - confidence: 置信度(0-1之间)
        - explanation: 详细的解释
        - suggestions: 具体的改进建议
        """
        return prompt

    def detect_td(self, method_code: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        检测单个方法的技术债务
        
        Args:
            method_code: 方法代码
            max_retries: 最大重试次数
            
        Returns:
            检测结果字典
        """
        prompt = self.create_td_detection_prompt(method_code)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的软件质量分析专家，擅长识别代码中的技术债务。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content.strip()
                
                # 尝试解析JSON响应
                try:
                    clean_text = re.sub(r"```[a-zA-Z]*", "", content)
                    clean_text = clean_text.replace("```", "").strip()
                    result = json.loads(clean_text)
                    return result.get("has_td", False)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，尝试提取关键信息
                    logger.warning(f"无法解析JSON响应，尝试提取信息: {content}")
                    return self._extract_info_from_text(content)
                    
            except Exception as e:
                logger.error(f"第{attempt + 1}次尝试失败: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return {
                        "has_td": False,
                        "td_types": [],
                        "confidence": 0.0,
                        "explanation": f"检测失败: {str(e)}",
                        "suggestions": []
                    }
        
        return {
            "has_td": False,
            "td_types": [],
            "confidence": 0.0,
            "explanation": "检测失败：超过最大重试次数",
            "suggestions": []
        }
    
    def _extract_info_from_text(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取信息（当JSON解析失败时的备用方案）
        
        Args:
            text: ChatGPT返回的文本
            
        Returns:
            提取的信息字典
        """
        # 简单的文本解析逻辑
        has_td = "true" in text.lower() or "是" in text or "存在" in text
        confidence = 0.5  # 默认置信度
        
        # 尝试提取置信度
        import re
        confidence_match = re.search(r'confidence["\']?\s*[:=]\s*([0-9.]+)', text.lower())
        if confidence_match:
            confidence = float(confidence_match.group(1))
        
        return {
            "has_td": has_td,
            "td_types": [],
            "confidence": confidence,
            "explanation": text,
            "suggestions": []
        }

    def batch_detect(self, csv_file: str, output_file: str = None, 
                    start_idx: int = 0, end_idx: int = None, 
                    delay: float = 1.0) -> pd.DataFrame:
        """
        批量检测CSV文件中的方法
        
        Args:
            csv_file: 输入CSV文件路径
            output_file: 输出文件路径
            start_idx: 开始索引
            end_idx: 结束索引
            delay: 请求间隔时间（秒）
            
        Returns:
            包含检测结果的DataFrame
        """
        logger.info(f"开始读取CSV文件: {csv_file}")
        df = pd.read_csv(csv_file)
        
        if end_idx is None:
            end_idx = len(df)
        
        df_subset = df.iloc[start_idx:end_idx].copy()
        logger.info(f"处理数据范围: {start_idx} - {end_idx}, 共 {len(df_subset)} 条记录")
        
        results = []
        
        for idx, row in df_subset.iterrows():
            logger.info(f"处理第 {idx + 1}/{len(df_subset)} 条记录")
            
            method_code = row['method_code']
            project = row['project']
            file_name = row['file_name']
            original_label = row['label']
            
            # 检测技术债务
            detection_result = self.detect_td(method_code)
            
            # 构建结果记录
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
            
            # 添加延迟避免API限制
            if delay > 0:
                time.sleep(delay)
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(results)
        
        # 保存结果
        if output_file:
            result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"结果已保存到: {output_file}")
        
        return result_df

def main():
    parser = argparse.ArgumentParser(description='使用ChatGPT API进行技术债务检测')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API密钥')
    parser.add_argument('--input_file', type=str, default='dataset/td-dataset.csv', 
                       help='输入CSV文件路径')
    parser.add_argument('--output_file', type=str, default='chatgpt_td_results.csv',
                       help='输出CSV文件路径')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help='使用的模型名称')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='开始处理的索引')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='结束处理的索引')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='请求间隔时间（秒）')
    parser.add_argument('--test_mode', action='store_true',
                       help='测试模式，只处理前10条记录')
    
    args = parser.parse_args()
    
    # 测试模式
    if args.test_mode:
        args.start_idx = 0
        args.end_idx = 10
        args.delay = 0.5
        logger.info("测试模式：只处理前10条记录")
    
    # 创建检测器
    detector = ChatGPTTDDetector(model=args.model)
    
    # 批量检测
    try:
        results = detector.batch_detect(
            csv_file=args.input_file,
            output_file=args.output_file,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            delay=args.delay
        )
        
        # 计算准确率
        if 'original_label' in results.columns:
            correct = (results['chatgpt_has_td'] == results['original_label']).sum()
            total = len(results)
            accuracy = correct / total if total > 0 else 0
            logger.info(f"准确率: {accuracy:.4f} ({correct}/{total})")
        
        logger.info("检测完成！")
        
    except Exception as e:
        logger.error(f"检测过程中出现错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()


