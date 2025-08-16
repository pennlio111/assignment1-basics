from tests.tokenizer import Tokenizer
import numpy as np
import os
from tqdm import tqdm
import time

DATA_PATH = './data/'

def main():
    tokenizer = Tokenizer.from_files(
        vocab_filepath=DATA_PATH + 'my_ts_vocab.pkl',
        merges_filepath=DATA_PATH + 'my_ts_merges.pkl',
        special_tokens=['<|endoftext|>']
    )
    
    def encode_dataset(dataset_path):
        """Encode dataset with progress bar and detailed statistics"""
        print(f"\n🔄 Encoding: {os.path.basename(dataset_path)}")
        
        # Get file size for progress estimation
        file_size = os.path.getsize(dataset_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"📁 File size: {file_size_mb:.1f} MB")
        
        # Read file with progress bar
        print("📖 Reading file...")
        with open(dataset_path, 'r') as f:
            dataset = f.read()
        
        print(f"📝 Text length: {len(dataset):,} characters")
        
        # Encode with progress bar
        print("🔤 Tokenizing...")
        start_time = time.time()
        
        # For very large files, we can process in chunks
        if len(dataset) > 1000000:  # 1M characters
            print("📊 Large file detected, processing in chunks...")
            chunk_size = 500000  # 500K characters per chunk
            chunks = [dataset[i:i+chunk_size] for i in range(0, len(dataset), chunk_size)]
            
            token_ids = []
            for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks", unit="chunk")):
                chunk_tokens = tokenizer.encode(chunk)
                token_ids.extend(chunk_tokens)
                
                # Update progress description
                tqdm.write(f"Chunk {i+1}/{len(chunks)}: {len(chunk_tokens):,} tokens")
        else:
            # Small file, process directly
            token_ids = tokenizer.encode(dataset)
        
        encoding_time = time.time() - start_time
        
        # Convert to numpy array
        print("💾 Converting to numpy array...")
        token_ids = np.array(token_ids, dtype=np.uint16)
        
        # Calculate statistics
        compression_ratio = len(dataset.encode('utf-8')) / len(token_ids)
        tokens_per_second = len(token_ids) / encoding_time
        
        print(f"\n📊 Encoding Statistics:")
        print(f"   ⏱️  Time: {encoding_time:.2f} seconds")
        print(f"   🚀 Speed: {tokens_per_second:,.0f} tokens/second")
        print(f"   📏 Input: {len(dataset):,} characters")
        print(f"   🎯 Output: {len(token_ids):,} tokens")
        print(f"   📦 Shape: {token_ids.shape}")
        print(f"   🗜️  Compression: {compression_ratio:.2f} bytes/token")
        
        # Save with progress bar
        output_path = DATA_PATH + os.path.basename(dataset_path) + '_token_ids.npy'
        print(f"\n💾 Saving to: {output_path}")
        
        np.save(output_path, token_ids)
        print(f"✅ Saved successfully!")
        
        return token_ids
    
    # encode_dataset(DATA_PATH + 'TinyStoriesV2-GPT4-valid.txt')
    # print("TinyStoriesV2-GPT4-valid.txt encoded")
    
    print("🚀 Starting TinyStoriesV2-GPT4-train.txt encoding...")
    print("=" * 60)
    
    try:
        token_ids = encode_dataset(DATA_PATH + 'TinyStoriesV2-GPT4-train.txt')
        print("\n🎉 Encoding completed successfully!")
        print(f"📊 Final result: {token_ids.shape[0]:,} tokens")
    except Exception as e:
        print(f"\n❌ Error during encoding: {e}")
        raise

if __name__ == "__main__":
    main()