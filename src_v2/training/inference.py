from src_v2.loader import load_module

_impl = load_module(__file__, "8_inference.py", "inference_impl")

load_for_inference = _impl.load_for_inference
predict_next_from_series = _impl.predict_next_from_series
