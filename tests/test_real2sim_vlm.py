"""Regression tests for Real2Sim VLM response handling."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from real2sim._stages.vlm import detect_objects
from real2sim.vlm_clients.client import parse_selection_results, query_model


class SelectionResultParsingTest(unittest.TestCase):
    def test_parses_literal_result_block(self) -> None:
        parsed = parse_selection_results('***\n[{"name": "box"}]\n***')

        self.assertEqual(parsed, [{"name": "box"}])

    def test_parses_xml_result_wrapper(self) -> None:
        parsed = parse_selection_results(
            '<result>\n[{"name": "box", "confidence": "high"}]\n</result>'
        )

        self.assertEqual(parsed, [{"name": "box", "confidence": "high"}])

    def test_parses_json_fence(self) -> None:
        parsed = parse_selection_results(
            '```json\n{"object_name": "wooden table"}\n```'
        )

        self.assertEqual(parsed, {"object_name": "wooden table"})

    def test_parses_bare_json_after_brief_prose(self) -> None:
        parsed = parse_selection_results(
            'Here is the result:\n[{"name": "wooden table"}]'
        )

        self.assertEqual(parsed, [{"name": "wooden table"}])

    def test_rejects_response_without_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "No valid JSON"):
            parse_selection_results("I cannot identify the scene.")


class ObjectDetectionPromptTest(unittest.TestCase):
    @patch("real2sim._stages.vlm.query_model")
    def test_repeats_output_contract_in_user_message(self, query_model) -> None:
        query_model.return_value = (
            "<result>\n"
            '[{"name": "wooden table", "description": "support", '
            '"confidence": "high"}]\n'
            "</result>"
        )

        objects, _ = detect_objects(
            Image.new("RGB", (4, 4)),
            model="test-model",
            reasoning_effort="high",
            accepted_confidences=("high",),
            max_objects=None,
        )

        self.assertEqual(objects[0]["name"], "wooden table")
        request = query_model.call_args.kwargs
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertIn("literal *** delimiters", request["messages"][0])
        self.assertIn("Do not use <result>", request["messages"][0])

    @patch("real2sim._stages.vlm.query_model")
    def test_keeps_valid_normalized_bbox_and_ignores_invalid_bbox(self, query_model) -> None:
        query_model.return_value = (
            "***["
            '{"name":"left box","description":"left","confidence":"high",'
            '"bbox_2d":[-5,100,450,1100]},'
            '{"name":"right box","description":"right","confidence":"high",'
            '"bbox_2d":[700,100,600,500]}'
            "]***"
        )

        objects, _ = detect_objects(
            Image.new("RGB", (8, 8)),
            model="test-model",
            accepted_confidences=("high",),
            max_objects=None,
        )

        self.assertEqual(objects[0]["bbox_2d"], [0.0, 100.0, 450.0, 1000.0])
        self.assertNotIn("bbox_2d", objects[1])


class QueryModelRequestTest(unittest.TestCase):
    @patch("real2sim.vlm_clients.client.OpenAI")
    @patch("real2sim.vlm_clients.client.get_backend_config")
    def test_sends_requested_reasoning_effort(
        self,
        get_backend_config,
        openai,
    ) -> None:
        get_backend_config.return_value = (
            "CUSTOM",
            "test-key",
            "https://example.invalid/v1",
            False,
        )
        openai.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

        response = query_model(
            messages=["inspect the scene"],
            selected_model="gpt-6-astra",
            reasoning_effort="high",
        )

        self.assertEqual(response, "ok")
        request = openai.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-6-astra")
        self.assertEqual(request["reasoning_effort"], "high")

    @patch("real2sim.vlm_clients.client.OpenAI")
    @patch("real2sim.vlm_clients.client.get_backend_config")
    def test_omits_reasoning_effort_when_disabled(
        self,
        get_backend_config,
        openai,
    ) -> None:
        get_backend_config.return_value = (
            "CUSTOM",
            "test-key",
            "https://example.invalid/v1",
            False,
        )
        openai.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

        query_model(
            messages=["inspect the scene"],
            selected_model="gpt-6-astra",
            reasoning_effort=None,
        )

        request = openai.return_value.chat.completions.create.call_args.kwargs
        self.assertNotIn("reasoning_effort", request)

if __name__ == "__main__":
    unittest.main()
