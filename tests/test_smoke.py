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

    def test_read_int_env_is_resilient_and_bounded(self):
        import os
        import main

        old = os.environ.get("DOCUAGENT_MAX_DOCS")
        try:
            os.environ["DOCUAGENT_MAX_DOCS"] = "not-a-number"
            self.assertEqual(main._read_int_env("DOCUAGENT_MAX_DOCS", 25, minimum=1, maximum=500), 25)

            os.environ["DOCUAGENT_MAX_DOCS"] = "-10"
            self.assertEqual(main._read_int_env("DOCUAGENT_MAX_DOCS", 25, minimum=1, maximum=500), 1)

            os.environ["DOCUAGENT_MAX_DOCS"] = "999999"
            self.assertEqual(main._read_int_env("DOCUAGENT_MAX_DOCS", 25, minimum=1, maximum=500), 500)
        finally:
            if old is None:
                os.environ.pop("DOCUAGENT_MAX_DOCS", None)
            else:
                os.environ["DOCUAGENT_MAX_DOCS"] = old

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

            with patch.object(main.HTTP_SESSION, "post") as post:
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

    def test_ollama_provider_for_chat_uses_local_endpoint(self):
        import os
        import importlib
        from unittest.mock import patch

        old_key = os.environ.get("UPSTAGE_API_KEY")
        old_demo = os.environ.get("DOCUAGENT_DEMO_MODE")
        old_mode = os.environ.get("DOCUAGENT_MODE")
        old_provider = os.environ.get("DOCUAGENT_LLM_PROVIDER")
        old_ollama_base = os.environ.get("OLLAMA_BASE_URL")
        old_ollama_model = os.environ.get("OLLAMA_MODEL")

        os.environ["UPSTAGE_API_KEY"] = "upstage_test_key_123456"
        os.environ["DOCUAGENT_LLM_PROVIDER"] = "ollama"
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
        os.environ["OLLAMA_MODEL"] = "llama3.2:latest"
        os.environ.pop("DOCUAGENT_DEMO_MODE", None)
        os.environ.pop("DOCUAGENT_MODE", None)

        class _FakeResponse:
            status_code = 200
            text = '{"message":{"content":"ollama answer"}}'

            @staticmethod
            def json():
                return {"message": {"content": "ollama answer"}}

        try:
            import main

            importlib.reload(main)

            with patch.object(main.HTTP_SESSION, "post", return_value=_FakeResponse()) as post:
                with patch.object(main, "_get_upstage_client", side_effect=AssertionError("Upstage should not be called")):
                    answer = main.call_solar_chat("system", "user")
                    self.assertEqual(answer, "ollama answer")
                    self.assertTrue(post.called)
        finally:
            if old_key is None:
                os.environ.pop("UPSTAGE_API_KEY", None)
            else:
                os.environ["UPSTAGE_API_KEY"] = old_key
            if old_demo is None:
                os.environ.pop("DOCUAGENT_DEMO_MODE", None)
            else:
                os.environ["DOCUAGENT_DEMO_MODE"] = old_demo
            if old_mode is None:
                os.environ.pop("DOCUAGENT_MODE", None)
            else:
                os.environ["DOCUAGENT_MODE"] = old_mode
            if old_provider is None:
                os.environ.pop("DOCUAGENT_LLM_PROVIDER", None)
            else:
                os.environ["DOCUAGENT_LLM_PROVIDER"] = old_provider
            if old_ollama_base is None:
                os.environ.pop("OLLAMA_BASE_URL", None)
            else:
                os.environ["OLLAMA_BASE_URL"] = old_ollama_base
            if old_ollama_model is None:
                os.environ.pop("OLLAMA_MODEL", None)
            else:
                os.environ["OLLAMA_MODEL"] = old_ollama_model


if __name__ == "__main__":
    unittest.main()
