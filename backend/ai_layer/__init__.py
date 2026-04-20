"""
AI Decision Layer — Layer 2 above the quant engine.

Receives structured prediction packets from Layer 1 (ml/predict.py),
applies eligibility rules, scores opportunities, and returns a clear
recommendation: STRONG BET / BET / SMALL BET / WATCHLIST / PASS / AVOID.

Architecture:
  packet_builder.py      — builds structured input packet from raw prediction
  rules_engine.py        — hard no-bet filters and eligibility gates
  scoring_engine.py      — weighted composite scoring (0–10)
  recommendation_service.py — orchestrates the above, returns final decision
  learning_engine.py     — logs decisions to DB for feedback loop
"""
