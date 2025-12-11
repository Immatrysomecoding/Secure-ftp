import os
import json
from datasets import load_dataset
import soundfile as sf

def setup_test_data():
    """Tạo đầy đủ test data"""
    
    print("🚀 BẮT ĐẦU SETUP TEST DATA")
    print("="*60)
    
    # Tạo cấu trúc thư mục
    folders = [
        "test_audio/vivos",
        "test_audio/common_voice", 
        "test_audio/manual",
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Tạo thư mục: {folder}")
    
    # 1. Download VIVOS
    print("\n📥 BƯỚC 1: Download VIVOS samples...")
    try:
        dataset = load_dataset("vivos", split="test")
        vivos_refs = []
        
        for i in range(min(20, len(dataset))):
            sample = dataset[i]
            audio = sample['audio']['array']
            sr = sample['audio']['sampling_rate']
            text = sample['sentence']
            
            filename = f"test_audio/vivos/vivos_{i:03d}.wav"
            sf.write(filename, audio, sr)
            
            vivos_refs.append({
                "file": filename,
                "text": text,
                "type": "clean_speech"
            })
            
            print(f"  ✅ {filename}")
        
        # Lưu references
        with open("test_audio/vivos/references.json", "w", encoding="utf-8") as f:
            json.dump(vivos_refs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Downloaded {len(vivos_refs)} VIVOS samples")
        
    except Exception as e:
        print(f"⚠️  Lỗi download VIVOS: {e}")
        vivos_refs = []
    
    # 2. Tạo synthetic samples
    print("\n🔊 BƯỚC 2: Tạo synthetic test cases...")
    try:
        from gtts import gTTS
        
        synthetic_cases = [
            ("xin chào đây là bài test", "test_001.wav", "clean"),
            ("tôi có meeting lúc ba giờ", "test_002.wav", "code_switching"),
            ("số điện thoại không một hai ba", "test_003.wav", "numbers"),
        ]
        
        synthetic_refs = []
        for text, filename, test_type in synthetic_cases:
            filepath = f"test_audio/manual/{filename}"
            tts = gTTS(text=text, lang='vi', slow=False)
            tts.save(filepath)
            
            synthetic_refs.append({
                "file": filepath,
                "text": text,
                "type": test_type
            })
            print(f"  ✅ {filepath}")
        
        with open("test_audio/manual/references.json", "w", encoding="utf-8") as f:
            json.dump(synthetic_refs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Tạo {len(synthetic_refs)} synthetic samples")
        
    except Exception as e:
        print(f"⚠️  Lỗi tạo synthetic: {e}")
        print("   (Chạy: pip install gtts)")
        synthetic_refs = []
    
    # 3. Tổng hợp references
    all_refs = vivos_refs + synthetic_refs
    
    with open("test_audio/all_references.json", "w", encoding="utf-8") as f:
        json.dump(all_refs, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "="*60)
    print("✅ SETUP HOÀN TẤT!")
    print("="*60)
    print(f"📁 Tổng số samples: {len(all_refs)}")
    print(f"   - VIVOS: {len(vivos_refs)}")
    print(f"   - Synthetic: {len(synthetic_refs)}")
    print(f"\n📄 References file: test_audio/all_references.json")
    print("\n🎯 BÂY GIỜ BẠN CÓ THỂ CHẠY TEST!")
    
    return all_refs

if __name__ == "__main__":
    refs = setup_test_data()