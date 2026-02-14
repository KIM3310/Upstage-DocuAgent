import unittest


class TestSmoke(unittest.TestCase):
    def test_import_main(self):
        # Import should not perform network calls.
        import main  # noqa: F401

    def test_extract_json_best_effort(self):
        import main

        payload = '{"a": 1, "b": {"c": 2}}'
        self.assertEqual(main._extract_json(payload), {"a": 1, "b": {"c": 2}})

        wrapped = "```json\n" + payload + "\n```"
        self.assertEqual(main._extract_json(wrapped), {"a": 1, "b": {"c": 2}})

        self.assertIsNone(main._extract_json("not json"))

    def test_demo_mode_does_not_call_network(self):
        import os
        import importlib
        from unittest.mock import patch

        old_key = os.environ.pop("UPSTAGE_API_KEY", None)
        old_mode = os.environ.get("DOCUAGENT_MODE")
        os.environ["DOCUAGENT_MODE"] = "demo"

        try:
            import main

            importlib.reload(main)

            with patch.object(main.requests, "post") as post:
                parsed = main.call_document_parse(b"hello world", "note.txt")
                self.assertIn("Demo Document Parse", parsed["markdown"])

                extracted = main.call_information_extract(
                    b"hello world",
                    "note.txt",
                    {"type": "object", "properties": {"title": {"type": "string"}}},
                )
                self.assertIn("title", extracted)

                edu = main.generate_edu_pack(
                    parsed_markdown=parsed["markdown"],
                    extracted_data=extracted,
                    audience="beginner",
                    goal="learn the basics",
                )
                for key in [
                    "learning_objectives",
                    "key_concepts",
                    "summary",
                    "quiz",
                    "flashcards",
                    "activities",
                ]:
                    self.assertIn(key, edu)

                # These should not require network access in demo mode.
                schema = main.auto_detect_schema(parsed["markdown"])
                self.assertIn("properties", schema)

                answer = main.call_solar_chat("system", "user")
                self.assertIn("Demo mode", answer)

                post.assert_not_called()
        finally:
            if old_key is not None:
                os.environ["UPSTAGE_API_KEY"] = old_key
            else:
                os.environ.pop("UPSTAGE_API_KEY", None)
            if old_mode is None:
                os.environ.pop("DOCUAGENT_MODE", None)
            else:
                os.environ["DOCUAGENT_MODE"] = old_mode


if __name__ == "__main__":
    unittest.main()
