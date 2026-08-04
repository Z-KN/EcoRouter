EcoRouter: Hardware-Aware Multi-Device GenAI Router

EcoRouter is a multi-device generative AI system that dynamically routes AI tasks across a
Snapdragon-powered Copilot+ PC, a Samsung Galaxy S25 powered by Snapdragon 8 Elite, and
Qualcomm Cloud AI 100 to deliver high-quality AI responses while minimizing hardware cost:
latency, battery drain, energy usage, thermal pressure, and cloud token cost

Today, most generative AI applications decide only which model to use. EcoRouter instead
decides where each part of an AI workflow should run. For example, a user can capture an
image, voice note, or text request on the Galaxy S25; the Snapdragon X/X Elite PC acts as the
central control surface and router; and Cloud AI 100 is used only when higher-quality reasoning
or generation is needed. The system continuously monitors device state such as phone battery,
PC load, network latency, estimated token cost, and response-quality requirements. Based on
these signals, it chooses whether to execute locally on the phone, locally on the PC, or remotely
on Cloud Al 100.

A sample demo use case is a "multimodal field assistant." The user captures a scene or
document on the phone and asks for an explanation, summary, checklist, or action plan.
EcoRouter decomposes the request into subtasks such as image/text extraction, privacy
screening, local summarization, quality checking, and final generation. Lightweight or privacy-
sensitive steps run on-device; more complex reasoning is escalated to the PC or Cloud Al 100
The PC dashboard visualizes routing decisions in real time, showing why each subtask was
placed on a specific device and reporting latency, estimated energy, battery impact, and cloud-
token usage

The project will use open-source software running natively on the Snapdragon-powered laptop,
including Python, FastAP| or WebSockets for device communication, a local dashboard using
Streamlit or React, and open-source small language/vision models for local inference. The
routing policy will combine rule-based constraints, lightweight classifiers, and cost/quality
scoring. Privacy will be treated as a guardrail: sensitive inputs can be summarized, redacted, or
kept local before any cloud call.
