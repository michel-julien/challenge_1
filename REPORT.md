# Challenge 1 - AI Security

## Cover page

**Title:** Analysis and security hardening of an AI-assisted access-control API  
**Team:** Team03  
**Verification date:** October 25, 2026  
**Scope:** local Python laboratory only

## 1. Introduction

The laboratory simulates a Flask API that receives an image, calls a facial
recognition model and an object-detection model, and then decides whether
physical access may be granted. The supplied facial model always returns
`authorized` with a confidence of `0.98`. The detector notably returns
`knife` with a confidence of `0.88`, making it possible to test a security
decision without using real data or systems.

The analysis follows the **PROVE -> DEFEND -> RETEST** cycle. The results in
this document are limited to the executions actually performed in the project
directory.

## 2. Scenario and security objective

The primary asset is the access decision. Secondary assets include the
confidentiality of the models and their structure, the integrity of the log
directory, service availability, and the confidentiality of diagnostics.
A secure decision must deny access when an image is invalid, a model fails, an
output is ambiguous, or a dangerous object is detected.

## 3. Architecture and attack surface

```text
[local client / curl / test_client]
              |
              | multipart image (untrusted)
 [Flask POST /api/access] -- T1 upload/HTTP boundary
              |
      decode + preprocess
              |
     [face model + object model]
              |
      fail-closed decision
              |
 [minimal JSON] + [logs/secure_uploads]
```

Untrusted inputs are the HTTP method, request size, filename, image bytes, and
model outputs. The original code, the models with known fingerprints, and the
decision policy are trusted assets.

## 4. Model preparation and compatibility

The installed dependencies were verified in `.venv`: Python 3.14.7,
NumPy 2.5.4, Flask 3.1.3, and OpenCV 5.0.0.93. Both pickle files are 55 bytes
long and reference `__main__.MockFaceRecognitionModel` or
`__main__.MockObjectDetectionModel`.

A direct import of `lab_access_api` produced the confirmed error
`AttributeError: module '__main__' has no attribute 'MockFaceRecognitionModel'`.
Running the file as a script is the compatibility behavior expected by the
generator. The test runner therefore loads the baseline under `__main__` to
reproduce the laboratory without modifying the models. The secure version
keeps local class definitions, verifies their SHA-256 hashes, and then uses a
`RestrictedModelUnpickler` limited to the two known classes.

## 5. Red Team analysis of the original code

The main weaknesses are:

1. `preprocess_image` calls `cv2.imdecode` without checking the buffer or the
   result, then returns a raw image without resizing, normalization, or a batch
   dimension.
2. The route calls `file.save()` and then `file.read()`. The stream is already
   at its end, so OpenCV receives empty bytes.
3. The policy looks only for the `weapon` label, while the model returns
   `knife`; the `0.15` threshold is not appropriate for a physical barrier.
4. The client-provided filename is concatenated with `./logs/`.
5. `pickle.load()` is unrestricted.
6. The response returns confidence, objects, model path, IP address, version,
   and log path.
7. There is no size limit, and `app.run(..., debug=True)` enables debugging at
   startup.
8. Exceptions are not converted into controlled errors, and the decision is
   not explicitly fail-closed.

The vulnerability matrix and impacts are summarized in the table above and in
the JSON evidence stored under `evidence/`.

## 6. Controlled Red Team tests

The fresh evidence is in `evidence/baseline_controlled_run_final.json`. The
older raw HTTP evidence is preserved in
`evidence/first_test.txt`, `evidence/first_error.txt`, and
`evidence/second_error.txt`.

Test, Action, Confirmed baseline result

A: POST without `image`, `400`, `{"error":"No image provided"}`
B: Synthetic 300x300 PNG, generic `500` with the test client; the earlier HTTP
evidence shows the OpenCV error caused by an empty buffer
C: Text file, the same OpenCV `500`, instead of a controlled rejection
Preprocessing: direct call on a valid PNG, `(300,300,3)` `uint8`; the model
rejects this shape
Isolated policy: temporarily inject the expected tensor to observe the rest of
the flow, `access=true` despite `knife:0.88`; internal details are present
Controlled path: filename `../logs/<unique name>.png`, without leaving the
laboratory; the client path appears in the response and the file remains under
the tested local directory

The policy instrumentation is explicitly isolated. It does not replace the
natural results of tests B/C; it makes it possible to reach the part of the
code that would otherwise be hidden by the stream bug.

## 7. Blue Team defenses

`lab_access_api_secure.py` applies the following fixes:

- request limit of 1 MiB and image limit of 900 KiB;
- single bounded in-memory read, validated decoding, and OpenCV errors
  converted into a generic `400`;
- resizing to 224x224, RGB conversion, `float32`, `[0,1]` normalization, and
  batch shape `(1,224,224,3)`;
- dangerous labels including `knife`, `gun`, `weapon`, weapon variants, and
  explosives; confirmed threshold `0.80`, while all ambiguity is still denied;
- minimum face confidence of `0.90`; unknown labels and malformed outputs are
  denied;
- UUID filename generated under `logs/secure_uploads`, with the resolved
  parent directory checked;
- minimal responses and diagnostics kept only on the server side;
- pinned SHA-256 hash and restricted unpickler for compatibility with the
  simulated model; in production, prefer a non-executable format or an
  isolated service;
- debugging explicitly disabled, loopback host by default, and generic error
  handling.

Each choice is commented directly in `lab_access_api_secure.py` and verified
by the tests under `tests/`.

## 8. Retest

The results are in `evidence/secure_controlled_retest_final_2.json`, and the
regression suite was executed with `unittest`: **9 tests, 9 passed**.

An initial malformed-request assertion expected an exception from the
multipart parser, whereas Werkzeug treated the payload as an empty form and
the route correctly returned `No image provided`. The initial failure is
preserved in `evidence/unit_test_output_final_9.txt`; the test was corrected
to explicitly exercise the HTTP `404` handler, and the corrected version
passed in `evidence/unit_test_output_final_9_fixed.txt`.

Test, Baseline, Secure version

A without image, `400`, `400`
B valid PNG, `500` before the model, `200`, `access:false` because `knife:0.88`
is denied
C text file, OpenCV `500` / trace in the HTTP evidence, `400`, generic message
without a traceback
D request > 1 MiB, no configured limit, `413`, request rejected
Preprocessing, `(300,300,3)` `uint8`, `(1,224,224,3)` `float32`, values `[0,1]`
Response, paths, IP, scores, and labels exposed in the isolated path, only
`{"access":false}`
Filename, client path used, UUID file directly under `logs/secure_uploads`
Models, unrestricted `pickle.load`, incompatible direct import, verified hashes
and approved classes loaded

Direct network startup was also attempted in the sandbox and failed with
`Operation not permitted`; this error is preserved in
`evidence/startup_probe_2026-10-25.txt`. The retests reported here are
therefore those of the local Flask test client, not a claimed successful
network listener.

## 9. Residual risks and recommendations

- Pickle remains intrinsically an executable format. The current control is
  appropriate for the two fixed laboratory artifacts, not for an arbitrary
  model.
- Embedded hashes should be replaced with a signed and protected chain of
  trust in a real deployment.
- Flask request size limits do not replace rate limits, quotas, timeouts,
  authentication, authorization, and monitoring.
- The simulated models demonstrate neither biometric accuracy nor robustness
  against adversarial transformations. Production deployment would require
  evaluation datasets, calibrated thresholds, human review, and dedicated
  security testing.
- The local demonstration log is not an immutable audit system; permissions,
  rotation, and retention must be added in production.

## 10. Conclusion

The PROVE phase confirmed input, stream, model-contract, dangerous-object
policy, loading, path, information-leakage, and configuration weaknesses. The
DEFEND phase fixes them in a separate file without overwriting
`lab_access_api.py`. The RETEST phase confirms denial of the `knife`, rejection
of invalid and oversized data, the tensor contract, and the reduced response.
Sandbox limitations and observed errors are preserved rather than presented as
successes.

## Local references

- `lab_access_api.py` and `generate_models.py`, laboratory-provided/adapted
  code.
- `lab_access_api_secure.py`, Blue Team implementation.
- `evidence/baseline_controlled_run_final.json` and
  `evidence/secure_controlled_retest_final_2.json`.
- `evidence/before_after_comparison_final.json`, machine-readable comparison
  of statuses.
- Materials provided in `files/`.
