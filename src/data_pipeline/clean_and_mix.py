import os,re
import logging
import hashlib
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from tokenizers import Tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

seen_hashes = set()

def normalize_text(example, lang):
    text = str(example['text'])
    if lang == 'zho':
        text = text.replace(r'$$\underline{}$$', '____')
        text = re.sub(r'[A-Za-z]+:[^\s]+', '', text)
        
        # save the simple formulas (like $x$, $y$, $E=mc^2$) by removing the surrounding $ signs, 
        # but replace the complex ones with [公式]
        text = re.sub(r'\$\$?([a-zA-Z0-9]+)\$\$?', r'\1', text)
        text = re.sub(r'\$\$?.*?\$\$?', '[公式]', text, flags=re.DOTALL)
        
        text = re.sub(r'\\[a-zA-Z]+(\{.*?\})?', '', text)
        text = text.replace('①', '1.').replace('②', '2.').replace('③', '3.').replace('④', '4.')
        text = re.sub(r'\\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # transform the speaker labels to a more standardized format (e.g., "老师:", "妈妈:", "小孩:") 
        # and ensure they are treated as part of the text rather than metadata
        text = text.replace('*TEACHER:', '老师:')
        text = text.replace('*MOTHER:', '妈妈:')
        text = text.replace('*TARGET_CHILD:', '小孩:')
        text = re.sub(r'\*[A-Z_]+:', '旁白:', text) 
        
        # transform the punctuation to a more consistent format (e.g., replace '.' with '。' in Chinese text) 
        # to help the tokenizer better learn the language-specific punctuation patterns
        text = text.replace('.', '。')
        
        # remove spaces between Chinese characters and punctuation to prevent the tokenizer from treating them as separate tokens,
        text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff。，！？、])', '', text)
        text = re.sub(r'(?<=[\u4e00-\u9fff。，！？、])\s+(?=[\u4e00-\u9fff])', '', text)
        
        # add newlines before speaker labels to help the model learn dialogue structure, 
        # but only for the main speakers to avoid over-segmentation
        text = text.replace(' 老师:', '\n老师:')
        text = text.replace(' 妈妈:', '\n妈妈:')
        text = text.replace(' 小孩:', '\n小孩:')
        
    example['text'] = text.strip()
    return example

def tag_aligned_corpus(example, lang):
    """
    Tag English-Chinese aligned corpus
    """
    is_aligned = False
    text = str(example['text'])
    
    if lang == 'zho':
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        alpha_chars = len(re.findall(r'[a-zA-Z]', text))
        
        # condition: both Chinese and English characters are greater than 15, and their ratio is within a reasonable range
        if chinese_chars > 15 and alpha_chars > 15:
            ratio = chinese_chars / alpha_chars
            if 0.2 < ratio < 5.0:
                is_aligned = True
                
    example['is_aligned'] = is_aligned
    return example

def is_unique(example):
    """
    Step 1: Uniqueness Filter using MD5 Hashing of normalized text
    """
    text = str(example['text'])
    
    # simplify the text by removing spaces and punctuation, and converting to lowercase, to create a more robust hash for deduplication
    simplified_text = re.sub(r'[^\w\u4e00-\u9fff]', '', text.lower())
    
    # if the simplified text is empty after removing noise, we can consider it as non-unique to filter out such cases
    if not simplified_text:
        return False
        
    # calculate MD5 Hash (more memory-efficient than storing strings)
    text_hash = hashlib.md5(simplified_text.encode('utf-8')).hexdigest()
    
    if text_hash in seen_hashes:
        return False
        
    seen_hashes.add(text_hash)
    return True

def deep_quality_filter(example, lang):
    """
    Step 2: Deep Data Quality Filter 
    """
    text = example['text']
    text_len = len(text)
    is_aligned = example.get('is_aligned', False)
    
    # 1. Length too short
    if lang == 'zho' and text_len < 3:
        return False
    if lang in ['eng', 'nld'] and text_len < 30:
        return False

    if text.count('[公式]') >= 3:
        return False

    alpha_chars_total = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', text))
        
    # Exempt aligned corpus from these checks
    if not is_aligned:
        if text_len > 0 and (alpha_chars_total / text_len) < 0.4:
            return False 
        if lang == 'zho':
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            if alpha_chars_total > 0 and (chinese_chars / alpha_chars_total) < 0.55:
                return False 
            
    # 4. Dutch Purity Check
    if lang == 'nld':
        words = set(text.lower().split())
        dutch_stopwords = {"de", "het", "een", "en", "van", "is", "dat", "in", "te", "op", "voor", "met", "zijn", "niet", "om"}
        english_stopwords = {"the", "and", "of", "to", "a", "that", "was", "he", "it", "with", "as", "his", "on", "be"}
        
        nld_count = len(words.intersection(dutch_stopwords))
        eng_count = len(words.intersection(english_stopwords))
        
        if len(words) >= 10 and nld_count == 0:
            return False
        if eng_count > nld_count:
            return False
    
    return True

def calculate_adjusted_tokens(target_tokens, lang):
    """
    Calculate actual allowed sampling tokens adjusted by Byte Premium
    """
    premiums = {'eng': 1.0, 'nld': 1.0516, 'zho': 0.93}
    # 公式: 實際抽取量 = 目標預算 / 溢價係數
    # Formula: Actual sampled = Target budget / Premium factor
    return int(target_tokens / premiums[lang])

def get_exact_row_count_for_budget(dataset, target_budget, lang, tokenizer):
    """
    Core conversion (Precise Version): Calculate exact token consumption by running actual BPE encoding.
    """
    accumulated_tokens = 0
    
    for idx, item in enumerate(dataset):
        text = item['text']
        
        if tokenizer is not None:
            # Precise calculation: Use BPE Tokenizer to convert text to ID array and count exact length
            encoded_ids = tokenizer.encode(text).ids
            exact_tokens = len(encoded_ids)
            accumulated_tokens += exact_tokens
        else:
            # 回退機制 (Fallback)：如果 Tokenizer 還沒訓練好，沿用先前的粗估邏輯
            if lang == 'zho':
                accumulated_tokens += len(text)
            else:
                accumulated_tokens += len(text.split())
        
        if accumulated_tokens >= target_budget:
            # 記錄誤差，你會發現精確版與粗估版會有差距！
            logging.debug(f"[{lang}] Actual sampled tokens: {accumulated_tokens:,} (Target: {int(target_budget):,})")
            return idx + 1 
            
    logging.warning(f"[{lang}] Warning: Reached end of dataset before hitting the target budget. Total tokens accumulated: {accumulated_tokens:,}")
    return len(dataset)


from datasets import DatasetDict

def prepare_stage_data(datasets, stage_name, ratios, total_budget, output_base_dir, tokenizer, vocab_name):
    """
    Create, sample, mix, shuffle, split train/val, and save the dataset for a specific stage.
    """
    logging.info(f"========== Preparing {stage_name} Data ==========")
    logging.info(f"--- Preparing {stage_name} for {vocab_name} ---")
    train_chunks = []
    
    for lang, ratio in ratios.items():
        # 1. 根據 Byte Premium 計算該語言的 Train Token 預算
        target_budget = total_budget * ratio
        actual_allowed_tokens = calculate_adjusted_tokens(target_budget, lang)
        
        dataset_lang = datasets[lang]
        # Use different seeds based on stage names to avoid identical sequences across stages
        stage_seed = hash(stage_name) % 10000
        shuffled_lang_ds = dataset_lang.shuffle(seed=stage_seed)
        
        # 2. compute the exact number of rows needed to meet the adjusted token budget using the precise BPE-based calculation
        if stage_name == "Stage_2_Alignment" and lang == 'zho' and "is_aligned" in shuffled_lang_ds.column_names:
            logging.info(f"[{lang}] Enabling Stage 2 Aligned Upsampling...")
            aligned_ds = shuffled_lang_ds.filter(lambda x: x['is_aligned'])
            regular_ds = shuffled_lang_ds.filter(lambda x: not x['is_aligned'])
            
            aligned_budget = actual_allowed_tokens * 0.50
            actual_aligned_rows = min(get_exact_row_count_for_budget(aligned_ds, aligned_budget, lang, tokenizer), len(aligned_ds))
            
            remaining_budget = actual_allowed_tokens - (aligned_budget * (actual_aligned_rows / max(1, actual_aligned_rows)))
            needed_regular_rows = get_exact_row_count_for_budget(regular_ds, remaining_budget, lang, tokenizer)
            
            train_aligned = aligned_ds.select(range(actual_aligned_rows))
            train_regular = regular_ds.select(range(needed_regular_rows))
            
            train_sampled = concatenate_datasets([train_aligned, train_regular]).shuffle(seed=42)
        else:
            needed_train_rows = get_exact_row_count_for_budget(shuffled_lang_ds, actual_allowed_tokens, lang, tokenizer)
            train_sampled = shuffled_lang_ds.select(range(needed_train_rows))

        if "language" in train_sampled.column_names:
            train_sampled = train_sampled.remove_columns("language")
        train_sampled = train_sampled.add_column("language", [lang] * len(train_sampled))
        train_chunks.append(train_sampled)
        
    mixed_train = concatenate_datasets(train_chunks).shuffle(seed=42)
    
    # Save only the train split here, as validation is handled globally!
    final_dataset = DatasetDict({'train': mixed_train})
    
    scale_folder = "processed_10M" if total_budget <= 10_000_000 else "processed_100M"
    save_path = os.path.join(output_base_dir, scale_folder, vocab_name, stage_name)
    
    final_dataset.save_to_disk(save_path)
    logging.info(f"✅ {stage_name} Train Set Mixed! Rows: {len(mixed_train):,}. Saved to: {save_path}\n")

def main():
    langs = ['eng', 'zho', 'nld']
    global_train_pool = {}
    
    # Define unified cache and validation paths
    pool_base_path = "data/train_pool"
    val_save_path = "data/global_validation"
    
    # Core Check: If all language train pools and the global validation folder exist, skip the cleaning process
    skip_cleaning = all(os.path.exists(os.path.join(pool_base_path, lang)) for lang in langs) and os.path.exists(val_save_path)
    
    if skip_cleaning:
        logging.info("🚀 ==================================================")
        logging.info("🚀 existing train_pool and global_validation detected!")
        logging.info("🚀 automatically loading cache, skipping time-consuming cleaning, deduplication, and splitting steps.")
        logging.info("🚀 ==================================================")
        
        # 直接從硬碟讀取純淨訓練池 (Directly load the clean train pool from disk)
        for lang in langs:
            pool_path = os.path.join(pool_base_path, lang)
            global_train_pool[lang] = load_from_disk(pool_path)
            logging.info(f"📦 loaded [{lang.upper()}] clean train pool, rows: {len(global_train_pool[lang]):,}")
            
    else:
        logging.info("⏳ ==================================================")
        logging.info("⏳ starting full data cleaning, deduplication, and validation split process...")
        logging.info("⏳ ==================================================")
        
        datasets = {}
        # 1. Original Load and Clean Data Logic
        for lang in langs:
            path = f"data/raw/{lang}_dataset"
            raw_ds = load_from_disk(path)

            train_split = raw_ds['train'] if isinstance(raw_ds, dict) else raw_ds
            original_size = len(train_split)

            logging.info(f"[{lang.upper()}] Normalization...")
            normalized_ds = train_split.map(lambda x: normalize_text(x, lang), load_from_cache_file=False)

            logging.info(f"[{lang.upper()}] Tagging Aligned Corpus...")
            tagged_ds = normalized_ds.map(lambda x: tag_aligned_corpus(x, lang), load_from_cache_file=False)
            
            logging.info(f"[{lang.upper()}] Filtering...")
            cleaned_ds = tagged_ds.filter(lambda x: deep_quality_filter(x, lang), load_from_cache_file=False)
            
            logging.info(f"[{lang.upper()}] Deduplicating...")
            seen_hashes.clear() 
            deduped_ds = cleaned_ds.filter(is_unique, load_from_cache_file=False)

            cleaned_size = len(deduped_ds)
            removed_size = original_size - cleaned_size
            removed_ratio = (removed_size / original_size) * 100 if original_size > 0 else 0
            
            datasets[lang] = DatasetDict({'train': deduped_ds})
            logging.info(f"{lang.upper()} Clean & Dedup Finished: {original_size:,} -> {cleaned_size:,} (Removed {removed_ratio:.2f}%)")

        # 2. Extract global validation and save caches on first run
        global_val_chunks = []
        for lang in langs:
            shuffled_ds = datasets[lang]['train'].shuffle(seed=42)
            val_size = min(5000, int(len(shuffled_ds) * 0.05))
            
            val_slice = shuffled_ds.select(range(val_size))
            train_slice = shuffled_ds.select(range(val_size, len(shuffled_ds)))
            
            if "language" in val_slice.column_names:
                val_slice = val_slice.remove_columns("language")
            val_slice = val_slice.add_column("language", [lang] * len(val_slice))
            
            global_val_chunks.append(val_slice)
            global_train_pool[lang] = train_slice

        # Save the global validation set to data/ root
        mixed_global_val = concatenate_datasets(global_val_chunks).shuffle(seed=42)
        mixed_global_val.save_to_disk(val_save_path)
        logging.info(f"🚨 Global Validation Set Created Accurately! Rows: {len(mixed_global_val):,}. Saved to: {val_save_path}")

        # Persist clean train pool to disk for future skipping
        os.makedirs(pool_base_path, exist_ok=True)
        for lang in langs:
            pool_save_path = os.path.join(pool_base_path, lang)
            global_train_pool[lang].save_to_disk(pool_save_path)
            logging.info(f"📦 Saved clean train pool for [{lang.upper()}] to: {pool_save_path}")

    # 3. Define total budget and experiment matrix 
    TOTAL_BUDGET = 10_000_000

    naive_baseline = {
        "Baseline_Naive": {'budget_ratio': 1.0, 'lang_ratios': {'eng': 0.334, 'zho': 0.333, 'nld': 0.333}}
    }
    static_baseline = {
        "Baseline_Static": {'budget_ratio': 1.0, 'lang_ratios': {'eng': 0.44, 'zho': 0.33, 'nld': 0.23}}
    }
    curriculum = {
        "Stage_1_Foundation": {'budget_ratio': 0.30, 'lang_ratios': {'eng': 0.50, 'zho': 0.25, 'nld': 0.25}},
        "Stage_2_Alignment": {'budget_ratio': 0.30, 'lang_ratios': {'eng': 0.33, 'zho': 0.33, 'nld': 0.34}},
        "Stage_3_HardBoosting": {'budget_ratio': 0.40, 'lang_ratios': {'eng': 0.20, 'zho': 0.40, 'nld': 0.40}}
    }

    vocab_configs = {
        "vocab_14k": "tokenizers/tokenizer_10M_14k.json",
        "vocab_16k": "tokenizers/tokenizer_10M_16k.json",
        "vocab_18k": "tokenizers/tokenizer_10M_18k.json",
        #"vocab_30k": "tokenizers/tokenizer_100M_30k.json",  
        #"vocab_32k": "tokenizers/tokenizer_100M_32k.json",
        #"vocab_34k": "tokenizers/tokenizer_100M_34k.json"
    }

    all_experiments = {**naive_baseline, **static_baseline, **curriculum}

    # 4. 生成混合實驗資料集 (Generate mixed data for ablation matrix)
    for vocab_name, tokenizer_path in vocab_configs.items():
        logging.info(f"\n========================================")
        logging.info(f"🚀 starting data generation for: {vocab_name}")
        logging.info(f"========================================")

        try:
            current_tokenizer = Tokenizer.from_file(tokenizer_path)
        except Exception:
            logging.warning(f"⚠️ Cannot find {tokenizer_path}, please check path.")
            continue

        for stage, config in all_experiments.items():
            stage_budget = TOTAL_BUDGET * config['budget_ratio']
            
            prepare_stage_data(
                datasets=global_train_pool,  
                stage_name=stage, 
                ratios=config['lang_ratios'],
                total_budget=stage_budget,
                output_base_dir="data",
                tokenizer=current_tokenizer,
                vocab_name=vocab_name       
            )          
if __name__ == "__main__":
    main()
