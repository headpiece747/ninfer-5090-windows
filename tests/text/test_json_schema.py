"""Independent JSON Schema oracle for the native compiler/matcher (no model inference).

Port change: every subprocess call states `encoding="utf-8"`. With `text=True` and no encoding,
Python uses the locale encoding, which is UTF-8 on the maintainer's Linux host and cp1252 on
Windows -- so this test raised UnicodeEncodeError on the first payload containing `你`, before the
probe ever ran. The payload is deliberately non-ASCII (`ensure_ascii=False`, CJK and emoji in the
string cases) and carries an embedded NUL, so the encoding has to be stated rather than inherited.
"""

import json
import os
from pathlib import Path
import subprocess
import unittest

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(os.environ.get("NINFER_SCHEMA_PROBE", ROOT / "build/tests/ninfer_json_schema_test"))


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class SchemaContracts(unittest.TestCase):
    def run_cases(self, cases):
        payload = "".join(compact({"schema": schema, "candidates": [compact(v) for v in values]}) + "\n"
                          for schema, values in cases)
        result = subprocess.run([str(PROBE), "--probe"], input=payload, text=True, encoding="utf-8",
                                capture_output=True, check=True, timeout=90)
        reports = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(reports), len(cases))
        for (schema, values), report in zip(cases, reports):
            with self.subTest(schema=schema):
                self.assertNotIn("error", report, report)
                validator = jsonschema.Draft202012Validator(schema)
                expected = [validator.is_valid(value) for value in values]
                self.assertEqual(report["accepted"], expected, (schema, values, report, expected))

    def test_values_against_independent_validator(self):
        scalars = [None, False, True, -2, -1, 0, 1, 2, 3, 7, 8, "", "a", "ab", [], {}]
        strings = ["", "a", "ab", "abc", "你", "你好", "😀", "\n", "\t", "\0", '"', "\\", "a\n", "\r\b\f"]
        objects = [{}, {"n": 1}, {"n": "1"}, {"n": 2, "s": "x"}, {"s": "x"}, {"n": 1, "extra": 0}]
        cases = [
            ({}, scalars),
            ({"type": "integer", "minimum": -1, "maximum": 2}, scalars),
            ({"type": ["integer", "null"], "exclusiveMinimum": 0, "exclusiveMaximum": 3}, scalars),
            ({"type": "string", "minLength": 1, "maxLength": 2}, strings),
            ({"type": "string", "pattern": "p"}, ["", "p", "apple", "pear", "a\np", "none", 'p"\\\n']),
            ({"type": "string", "pattern": "^a(b|c){1,2}$"}, ["ab", "abc", "acb", "a", "abbc", "zab", "abz"]),
            ({"type": "string", "pattern": "\\n"}, ["", "\n", "a\nb", "\\n"]),
            # jsonschema uses Python re, whose dot differs from ECMAScript on CR/U+2028/U+2029.
            # The native test checks those separately against the ECMAScript line terminator set.
            ({"type": "string", "pattern": "^a.b$"}, ["acb", "a\nb", "a\tb"]),
            ({"type": "string", "pattern": r"^[\^\[]$"}, ["^", "[", "x", "]"]),
            ({"type": "string", "pattern": r"^\s$"}, [" ", "\n", "\u1680", "\u2003", "a"]),
            ({"type": "string", "pattern": r"^\S$"}, [" ", "\n", "\u1680", "\u2003", "a"]),
            ({"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2},
             [[], [1], [1, 2], [1, 2, 3], ["x"], [None]]),
            ({"type": "object", "properties": {"n": {"type": "integer"}, "s": {"type": "string"}},
              "required": ["n"], "additionalProperties": False}, objects),
            ({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}, objects),
            ({"type": "object", "additionalProperties": {"type": "integer"}}, [{}, {"a": 1}, {"a": "x"}, {"a": 1, "b": 2}]),
            ({"enum": [1, "x", None], "type": "string"}, scalars + ["x"]),
            ({"anyOf": [{"type": "null"}, {"type": "integer", "minimum": 1}]}, scalars),
            ({"oneOf": [{"type": "string"}, {"type": "integer"}]}, scalars),
            ({"allOf": [{"type": "string"}]}, scalars),
            ({"$defs": {"a/b": {"type": "integer"}}, "$ref": "#/$defs/a~1b"}, scalars),
            ({"$defs": {"x": {"anyOf": [{"type": "integer"}, {"type": "string"}]}}, "$ref": "#/$defs/x/anyOf/1"}, scalars),
            ({"properties": {"n": {"type": "integer"}}}, scalars + [{"n": 1}, {"n": "bad"}]),
            ({"type": "array", "items": False}, [[], [0]]),
            ({"type": "object", "properties": {"bad": False}, "additionalProperties": False}, [{}, {"bad": 0}]),
            ({"type": ["integer", "null"], "minimum": 2, "maximum": 1}, [None, 1, 2]),
            ({"type": "object", "properties": {"bad": {"type": "string", "minLength": 2, "maxLength": 1}},
              "additionalProperties": False}, [{}, {"bad": "a"}, {"bad": "ab"}]),
            ({"type": "array", "items": {"type": "integer", "minimum": 2, "maximum": 1}}, [[], [1], [2]]),
            ({"type": "object", "properties": {
                "bad": {"type": "integer", "const": "x", "$defs": {"value": {"type": "integer"}}},
                "good": {"$ref": "#/properties/bad/$defs/value"}}, "additionalProperties": False},
             [{}, {"good": 1}, {"good": "x"}, {"bad": "x"}]),
            ({"type": "object", "properties": {"a": {"const": {"description": "a"}}, "b": {"const": {"description": "b"}}},
              "required": ["a", "b"], "additionalProperties": False},
             [{"a": {"description": "a"}, "b": {"description": "b"}}, {"a": {"description": "a"}, "b": {"description": "a"}}]),
        ]
        for value in [-(2**63), -(2**53)-1, 2**53+1, 2**63-1]:
            cases.append(({"type": "integer", "minimum": value, "maximum": value},
                          [value-1, value, value+1]))
        tree = {"$defs": {"node": {"anyOf": [{"type": "null"}, {"type": "object", "properties": {
            "value": {"type": "integer"}, "next": {"$ref": "#/$defs/node"}}, "required": ["value", "next"],
            "additionalProperties": False}]}}, "$ref": "#/$defs/node"}
        cases.append((tree, [None, {"value": 1, "next": None}, {"value": 1, "next": {"value": 2, "next": None}}, {"value": "bad", "next": None}]))
        self.run_cases(cases)

    def test_additional_keys_cannot_override_declared_values(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
        candidates = ['{"n":1,"n":"bad"}', '{"n":1,"\\u006e":"bad"}', '{"n":1,"x":"good"}']
        result = subprocess.run(
            [str(PROBE), "--probe"],
            input=compact({"schema": schema, "candidates": candidates}) + "\n",
            text=True, encoding="utf-8", capture_output=True, check=True, timeout=30,
        )
        self.assertEqual(json.loads(result.stdout)["accepted"], [False, False, True])

    def test_rejections_and_diagnostics(self):
        cases = [({"type": "string", "format": "email"}, "/format"),
                 ({"type": "array", "uniqueItems": True}, "/uniqueItems"),
                 ({"type": "string", "pattern": "x", "maxLength": 5}, "/pattern"),
                 ({"type": "object", "properties": {"x": {"type": "number", "minimum": 1}}}, "/properties/x/minimum"),
                 ({"$ref": "https://example.com/schema"}, "/$ref"),
                 ({"type": "string", "pattern": r"[\q]"}, "/pattern"),
                 ({"type": "string", "pattern": r"\xZZ"}, "/pattern"),
                 ({"type": "integer", "exclusiveMinimum": 2**63-1}, "/exclusiveMinimum"),
                 ({"const": {"large": 2**63+1}}, "/const/large"),
                 ({"const": -(2**63)-1}, "/const"),
                 ({"type": "string", "minLength": -1}, "/minLength")]
        result = subprocess.run(
            [str(PROBE), "--probe"],
            input="".join(compact({"schema": s, "candidates": []}) + "\n" for s, _ in cases),
            text=True, encoding="utf-8", capture_output=True, check=True, timeout=30,
        )
        self.assertEqual(len(result.stdout.splitlines()), len(cases))
        for report, (schema, pointer) in zip(result.stdout.splitlines(), cases):
            with self.subTest(schema=schema):
                data = json.loads(report)
                self.assertIn("error", data)
                self.assertEqual(data["pointer"], pointer)

    def test_empty_languages_fail_before_generation(self):
        schemas = [False, {"const": "x", "type": "integer"}, {"$ref": "#"},
                   {"type": "object", "properties": {"x": False}, "required": ["x"]},
                   {"anyOf": [False, {"type": "integer", "minimum": 2, "maximum": 1}]}]
        result = subprocess.run(
            [str(PROBE), "--probe"],
            input="".join(compact({"schema": s, "candidates": []}) + "\n" for s in schemas),
            text=True, encoding="utf-8", capture_output=True, check=True, timeout=30,
        )
        self.assertEqual(len(result.stdout.splitlines()), len(schemas))
        for schema, line in zip(schemas, result.stdout.splitlines()):
            with self.subTest(schema=schema):
                self.assertIn("error", json.loads(line))


if __name__ == "__main__":
    unittest.main()
