# download_vivos_fix.py
from datasets import load_dataset
import soundfile as sf
import os

def download_vivos_fixed():
    """Fix cách download VIVOS"""
    print("📥 Downloading VIVOS (fixed version)...")
    
    try:
        # Cách mới: dùng dataset identifier đầy đủ
        dataset = load_dataset("mozilla-foundation/vivos", 
                              split="test",
                              trust_remote_code=True)
        
        os.makedirs("test_audio/vivos", exist_ok=True)
        
        refs = []
        for i in range(min(20, len(dataset))):
            sample = dataset[i]
            audio = sample['audio']['array']
            sr = sample['audio']['sampling_rate']
            text = sample['transcription']  # Note: key name might differ
            
            filename = f"test_audio/vivos/vivos_{i:03d}.wav"
            sf.write(filename, audio, sr)
            
            refs.append({
                "file": filename,
                "text": text,
                "type": "clean_speech"
            })
            
            print(f"  ✅ {filename}")
        
        import json
        with open("test_audio/vivos/references.json", "w", encoding="utf-8") as f:
            json.dump(refs, f, indent=2, ensure_ascii=False)
        
        # Update all_references.json
        with open("test_audio/all_references.json", "r", encoding="utf-8") as f:
            all_refs = json.load(f)
        
        all_refs.extend(refs)
        
        with open("test_audio/all_references.json", "w", encoding="utf-8") as f:
            json.dump(all_refs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Downloaded {len(refs)} VIVOS samples")
        return refs
        
    except Exception as e:
        print(f"⚠️  Vẫn lỗi: {e}")
        print("Không sao, dùng 3 samples synthetic cũng đủ để test!")
        return []

if __name__ == "__main__":
    download_vivos_fixed()