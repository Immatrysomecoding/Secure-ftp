# test_whisper_simple.py
import whisper
import json
from jiwer import wer

# Load model
print("Loading Whisper...")
model = whisper.load_model("base")

# Load references
with open("test_audio/all_references.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"\n🧪 Testing {len(test_data)} files...\n")

# Test each file
results = []
for i, data in enumerate(test_data, 1):
    print(f"[{i}/{len(test_data)}] Testing: {data['file']}")
    
    # Transcribe
    result = model.transcribe(data['file'], language="vi")
    predicted = result['text']
    
    # Calculate WER
    error = wer(data['text'], predicted)
    
    # Print result
    print(f"  Reference:  {data['text']}")
    print(f"  Predicted:  {predicted}")
    print(f"  WER: {error:.2%}")
    print()
    
    results.append({
        "file": data['file'],
        "reference": data['text'],
        "predicted": predicted,
        "wer": error
    })

# Save results
with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Summary
avg_wer = sum(r['wer'] for r in results) / len(results)
print("="*60)
print(f"✅ DONE! Average WER: {avg_wer:.2%}")
print(f"📄 Results saved to: test_results.json")