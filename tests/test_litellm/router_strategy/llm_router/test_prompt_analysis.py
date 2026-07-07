"""Tests for the content-aware prompt analysis used by the LLM router."""

import pytest

from litellm.router_strategy.llm_router.prompt_analysis import (
    analyze_prompt,
    messages_have_images,
)


class TestCategoryDetection:
    def test_coding_prompt_scores_coding_highest(self):
        a = analyze_prompt("debug this python function and fix the algorithm bug in my api code")
        assert a.coding > a.reasoning
        assert a.coding > a.creativity
        assert a.coding > a.factual

    def test_reasoning_prompt_scores_reasoning_highest(self):
        a = analyze_prompt("calculate the velocity then prove the result using this logic sequence")
        assert a.reasoning > a.coding
        assert a.reasoning > a.creativity

    def test_creative_prompt_scores_creativity_highest(self):
        a = analyze_prompt("write a short fantasy story about a character in an imagined scene")
        assert a.creativity > a.coding
        assert a.creativity > a.reasoning

    def test_factual_prompt_scores_factual_highest(self):
        a = analyze_prompt("what is the capital of france and explain its history")
        assert a.factual > a.coding
        assert a.factual > a.creativity

    def test_code_block_forces_max_coding(self):
        a = analyze_prompt("here is context\n```\nx = 1\n```\nplease review")
        assert a.coding == 1.0

    def test_empty_of_signal_defaults_to_factual(self):
        a = analyze_prompt("hi")
        assert a.factual == 0.5
        assert a.reasoning == 0.0
        assert a.coding == 0.0
        assert a.creativity == 0.0
        assert a.complexity == 0.0


class TestComplexity:
    def test_long_prompt_is_more_complex_than_short(self):
        short = analyze_prompt("write code")
        long = analyze_prompt("write code " + "x" * 1600)
        assert long.complexity > short.complexity

    def test_complexity_keywords_raise_complexity(self):
        plain = analyze_prompt("explain this topic")
        hard = analyze_prompt("explain this distributed scalable production-ready architecture in depth")
        assert hard.complexity > plain.complexity

    def test_complexity_capped_at_one(self):
        a = analyze_prompt(
            "advanced expert comprehensive optimized distributed scalable architecture "
            "recursive async concurrent generics unit test data structure graph tree " * 3
        )
        assert a.complexity == 1.0


class TestCapabilityFlags:
    def test_images_set_vision(self):
        assert analyze_prompt("look", has_images=True).vision == 1.0
        assert analyze_prompt("look", has_images=False).vision == 0.0

    def test_tools_set_tools_and_bump_complexity(self):
        without = analyze_prompt("call something")
        with_tools = analyze_prompt("call something", has_tools=True)
        assert with_tools.tools == 1.0
        assert without.tools == 0.0
        assert with_tools.complexity > without.complexity

    def test_json_mode_bumps_coding(self):
        without = analyze_prompt("return an object")
        with_json = analyze_prompt("return an object", has_json_mode=True)
        assert with_json.coding > without.coding


class TestMessagesHaveImages:
    def test_detects_image_url_part(self):
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
        ]
        assert messages_have_images(messages) is True

    def test_plain_text_messages_have_no_images(self):
        assert messages_have_images([{"role": "user", "content": "hello"}]) is False

    def test_mixed_parts_without_image(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert messages_have_images(messages) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
