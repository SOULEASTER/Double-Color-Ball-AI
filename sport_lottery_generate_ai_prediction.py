# -*- coding: utf-8 -*-
"""
大乐透 AI 预测自动生成脚本
自动调用 AI 模型生成下期预测数据
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

# 尝试导入并加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


BASE_URL = os.environ.get("AI_BASE_URL") or "https://aihubmix.com/v1"
API_KEY = os.environ.get("AI_API_KEY")
if not API_KEY:
    print("❌ 请设置环境变量 AI_API_KEY")
    sys.exit(1)

MODELS = [
    {"id": "gpt-4.1-free", "name": "GPT-4.1", "model_id": "dlt_gpt_4.1"},
    {"id": "gemini-3-flash-preview-free", "name": "Gemini-3-flash-preview", "model_id": "dlt_gemini_3_flash"},
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOTTERY_HISTORY_FILE = os.path.join(SCRIPT_DIR, "data", "sports_lottery_data.json")
AI_PREDICTIONS_FILE = os.path.join(SCRIPT_DIR, "data", "sport_lottery_ai_predictions.json")
PREDICTIONS_HISTORY_FILE = os.path.join(SCRIPT_DIR, "data", "sport_lottery_predictions_history.json")
PROMPT_FILE = os.path.join(SCRIPT_DIR, "doc", "sport_lottery_prompt2.0.md")


def load_prompt_template() -> str:
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        return f.read()


def normalize_blue_balls(value: Any) -> List[str]:
    if isinstance(value, list):
        return sorted(str(item).zfill(2) for item in value)
    if isinstance(value, str) and value:
        return [str(value).zfill(2)]
    return []


def normalize_draw(draw: Dict[str, Any]) -> Dict[str, Any]:
    red_balls = sorted(str(item).zfill(2) for item in draw.get("red_balls", []))
    blue_balls = normalize_blue_balls(draw.get("blue_balls", draw.get("blue_ball")))
    return {
        **draw,
        "red_balls": red_balls,
        "blue_balls": blue_balls,
        "blue_ball": blue_balls[0] if blue_balls else None,
    }


def load_lottery_history() -> Dict[str, Any]:
    with open(LOTTERY_HISTORY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {
        **data,
        "data": [normalize_draw(draw) for draw in data.get("data", [])]
    }


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def extract_json_from_response(response_text: str) -> str:
    text = response_text.strip()
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        return text[start:end].strip()
    return text


def call_ai_model(client: OpenAI, model_config: Dict[str, str], prompt: str) -> Dict[str, Any]:
    response_text = ""
    try:
        print(f"  ⏳ 正在调用 {model_config['name']} 模型...")
        response = client.chat.completions.create(
            model=model_config['id'],
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的彩票数据分析师，擅长基于历史数据进行模式分析和预测。请严格按照要求返回 JSON 格式数据，不要有任何额外的解释或说明。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8
        )
        response_text = (response.choices[0].message.content or "").strip()
        prediction_data = json.loads(extract_json_from_response(response_text))
        print(f"  ✅ {model_config['name']} 预测成功")
        return prediction_data
    except json.JSONDecodeError as e:
        print(f"  ❌ {model_config['name']} JSON 解析失败: {str(e)}")
        print(f"  原始响应前500字符:\n{response_text[:500]}")
        raise
    except Exception as e:
        print(f"  ❌ {model_config['name']} 调用失败")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {str(e)}")
        raise


def validate_prediction(prediction: Dict[str, Any]) -> bool:
    try:
        required_fields = ["prediction_date", "target_period", "model_id", "model_name", "predictions"]
        for field in required_fields:
            if field not in prediction:
                print(f"    ⚠️  缺少字段: {field}")
                return False

        if len(prediction["predictions"]) != 5:
            print(f"    ⚠️  预测组数量不正确: {len(prediction['predictions'])}")
            return False

        for group in prediction["predictions"]:
            red_balls = group.get("red_balls", [])
            blue_balls = group.get("blue_balls", group.get("blue_ball", []))
            if len(red_balls) != 5:
                print(f"    ⚠️  前区数量不正确: {len(red_balls)}")
                return False
            if red_balls != sorted(red_balls):
                print(f"    ⚠️  前区未排序: {red_balls}")
                return False

            normalized_blue = normalize_blue_balls(blue_balls)
            if len(normalized_blue) != 2:
                print(f"    ⚠️  后区数量不正确: {len(normalized_blue)}")
                return False
            if normalized_blue != sorted(normalized_blue):
                print(f"    ⚠️  后区未排序: {normalized_blue}")
                return False
        return True
    except Exception as e:
        print(f"    ⚠️  验证出错: {str(e)}")
        return False


def normalize_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
    normalized_predictions = []
    for group in prediction.get("predictions", []):
        blue_balls = normalize_blue_balls(group.get("blue_balls", group.get("blue_ball")))
        normalized_predictions.append({
            **group,
            "red_balls": sorted(str(item).zfill(2) for item in group.get("red_balls", [])),
            "blue_balls": blue_balls,
            "blue_ball": blue_balls[0] if blue_balls else None,
        })

    return {
        **prediction,
        "predictions": normalized_predictions
    }


def calculate_hit_result(prediction_group: Dict[str, Any], actual_result: Dict[str, Any]) -> Dict[str, Any]:
    red_hits = [b for b in prediction_group["red_balls"] if b in actual_result["red_balls"]]
    blue_hits = [b for b in prediction_group["blue_balls"] if b in actual_result["blue_balls"]]
    return {
        "red_hits": red_hits,
        "red_hit_count": len(red_hits),
        "blue_hits": blue_hits,
        "blue_hit_count": len(blue_hits),
        "total_hits": len(red_hits) + len(blue_hits)
    }


def archive_old_prediction(lottery_data: Dict[str, Any]) -> None:
    try:
        if not os.path.exists(AI_PREDICTIONS_FILE):
            print("  ℹ️  没有旧预测需要归档\n")
            return

        with open(AI_PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
            old_predictions = json.load(f)

        old_target_period = old_predictions.get("target_period")
        if not old_target_period:
            print("  ⚠️  旧预测文件格式异常，跳过归档\n")
            return

        latest_period = lottery_data.get("data", [{}])[0].get("period")
        if not latest_period or int(old_target_period) > int(latest_period):
            print(f"  ℹ️  旧预测期号 {old_target_period} 尚未开奖，无需归档\n")
            return

        actual_result: Optional[Dict[str, Any]] = None
        for draw in lottery_data.get("data", []):
            if draw.get("period") == old_target_period:
                actual_result = draw
                break

        if not actual_result:
            print(f"  ⚠️  找不到期号 {old_target_period} 的开奖结果，跳过归档\n")
            return

        history_data = {"历史预测记录": "本文件保存已开奖期号的大乐透 AI 预测数据，用于对比和统计", "predictions_history": []}
        if os.path.exists(PREDICTIONS_HISTORY_FILE):
            with open(PREDICTIONS_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history_data = json.load(f)

        existing_record = next((r for r in history_data["predictions_history"] if r["target_period"] == old_target_period), None)
        if existing_record:
            print(f"  ℹ️  期号 {old_target_period} 已存在于历史记录中\n")
            return

        models_with_hits = []
        for model_data in old_predictions.get("models", []):
            predictions_with_hits = []
            for pred_group in model_data.get("predictions", []):
                normalized_group = normalize_prediction({"predictions": [pred_group]}).get("predictions", [pred_group])[0]
                pred_with_hit = normalized_group.copy()
                pred_with_hit["hit_result"] = calculate_hit_result(normalized_group, actual_result)
                predictions_with_hits.append(pred_with_hit)

            best_pred = max(predictions_with_hits, key=lambda p: p["hit_result"]["total_hits"])
            models_with_hits.append({
                "model_id": model_data.get("model_id"),
                "model_name": model_data.get("model_name"),
                "predictions": predictions_with_hits,
                "best_group": best_pred["group_id"],
                "best_hit_count": best_pred["hit_result"]["total_hits"]
            })

        new_record = {
            "prediction_date": old_predictions.get("prediction_date"),
            "target_period": old_target_period,
            "actual_result": actual_result,
            "models": models_with_hits
        }
        history_data["predictions_history"].insert(0, new_record)

        with open(PREDICTIONS_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)

        print(f"  ✅ 已将期号 {old_target_period} 的预测归档到历史记录")
        print(f"  📊 归档模型数: {len(models_with_hits)}\n")
    except Exception as e:
        print(f"  ⚠️  归档旧预测时出错: {str(e)}")
        print("  继续生成新预测...\n")


def generate_predictions() -> Optional[Dict[str, Any]]:
    print("\n" + "=" * 50)
    print("🤖 大乐透 AI 预测自动生成")
    print("=" * 50 + "\n")

    prompt_template = load_prompt_template()
    lottery_data = load_lottery_history()
    archive_old_prediction(lottery_data)

    next_draw = lottery_data.get("next_draw", {})
    target_period = next_draw.get("next_period", "")
    target_date = next_draw.get("next_date_display", "")
    if not target_period:
        print("❌ 无法获取下期期号信息")
        return None

    history_data = lottery_data.get("data", [])[:30]
    history_json = json.dumps(history_data, ensure_ascii=False, indent=2)
    prediction_date = datetime.now().strftime("%Y-%m-%d")

    print(f"🎯 目标期号: {target_period}")
    print(f"📅 开奖日期: {target_date}")
    print(f"📝 历史数据: 最近 {len(history_data)} 期")
    print(f"📅 预测日期: {prediction_date}\n")

    client = get_openai_client()
    all_predictions = []

    for model_config in MODELS:
        try:
            prompt = prompt_template.format(
                target_period=target_period,
                target_date=target_date,
                lottery_history=history_json,
                prediction_date=prediction_date,
                model_id=model_config['model_id'],
                model_name=model_config['name']
            )
            prediction = normalize_prediction(call_ai_model(client, model_config, prompt))
            if validate_prediction(prediction):
                all_predictions.append(prediction)
                print("  ✓ 验证通过\n")
            else:
                print("  ✗ 验证失败，跳过该模型\n")
        except Exception as e:
            print(f"  ✗ 处理 {model_config['name']} 时失败: {str(e)}\n")
            continue

    if not all_predictions:
        print("❌ 没有成功生成任何预测")
        return None

    return {
        "prediction_date": prediction_date,
        "target_period": target_period,
        "models": all_predictions
    }


def save_predictions(predictions: Dict[str, Any]) -> None:
    print("💾 保存预测数据...")
    if os.path.exists(AI_PREDICTIONS_FILE):
        backup_file = AI_PREDICTIONS_FILE.replace(".json", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(AI_PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 已创建备份: {os.path.basename(backup_file)}")

    with open(AI_PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 已保存到: {AI_PREDICTIONS_FILE}\n")


def main() -> None:
    predictions = generate_predictions()
    if predictions:
        save_predictions(predictions)
        print("=" * 50)
        print("🎉 大乐透预测生成完成！")
        print("=" * 50 + "\n")
        print("📋 预测摘要:")
        print(f"  期号: {predictions['target_period']}")
        print(f"  日期: {predictions['prediction_date']}")
        print(f"  模型数量: {len(predictions['models'])}")
        for model in predictions['models']:
            print(f"    - {model['model_name']}")
    else:
        print("❌ 大乐透预测生成失败")


if __name__ == "__main__":
    main()
