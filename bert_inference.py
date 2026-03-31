import torch
import torch.nn as nn
from tqdm import tqdm
import time
import csv
from pathlib import Path
import argparse
import datasets
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from brain.work.arch.config import Config, ArchType, DataTypeBert
import brain.work.arch.bert as bert
import brain.work.arch.util as archutil
import brain.work.tmpl.util as tmplutil

class BertWork:
    """
    ArchType, Config, DataType을 파라미터로 받는 최종 워크로드 클래스.
    """
    def __init__(self, arch_type: ArchType, data_type: DataTypeBert, config: Config, device):
        self.ctx_window_enc = config.ctx_window_enc

        self.arch_type = arch_type
        self.data_type = data_type
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
    
        # self._setup_model()

    def _setup_model(self, num_labels):
        """arch_type과 config 객체를 기반으로 모델을 준비합니다."""
        # -------------------- model setting
        if self.arch_type == ArchType.BERT:
            ModelClass = bert.BertModel
        else:
            raise ValueError(f"지원하지 않는 아키텍처 타입입니다: {self.arch_type.name}")

        try:
            self.model = ModelClass(
                bert_config=self.config, 
                use_classifier=True,
                num_classes=num_labels
            ).to(self.device)
            self.model.eval()
            print("✅ 모델이 성공적으로 GPU에 로드되었습니다.")
        except torch.cuda.OutOfMemoryError:
            print("cuda OOM")
            self.model = None # 진행 불가 상태임을 명시
        except Exception as e:
            print(f"❌ 모델 설정 중 예상치 못한 오류 발생: {e}")
            self.model = None

    def _load_data(self, path, name):
        """Load our dataset for inference.

        The size of a work dataset must be bigger than batch size for proper
        experiments, and we carefully set that the dataset size of work is
        `brain.work.tmpl.util.INF_DATASET_SIZE`. Since the default test split
        from the Hugging Face Datasets library may be smaller than we require,
        we use part of the training split as our work(inference) dataset.
        """
        return datasets.load_dataset(path, name, split=f"train[:{tmplutil.INF_DATASET_SIZE}]")

    # BERTWork 클래스 내의 _prepare_data 메서드
    def _prepare_data(self):
        """
        DataTypeBert 이름은 glue 의 태스크를 의미하므로
        각 태스크마다 키를 통해 로드하고 토큰화하여 DataLoader를 생성합니다.
        """
        # -------------------- toeknizer setting
        if self.arch_type == ArchType.BERT :
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                # Llama 토크나이저는 pad_token이 없는 경우가 많아 eos_token으로 설정
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                print(f"오류: 토크나이저 다운로드에 실패했습니다. - {e}")
                self.tokenizer = None

        if self.tokenizer is None:
            raise ValueError("오류: 데이터 처리를 위해 tokenizer가 필요합니다.")

        task_name = self.data_type.name.lower()
        #glue task data structure
        task_to_keys = {
            "cola": {"keys": ("sentence", None), "num_labels": 2},
            "mnli": {"keys": ("premise", "hypothesis"), "num_labels": 3},
            "mrpc": {"keys": ("sentence1", "sentence2"), "num_labels": 2},
            "qnli": {"keys": ("question", "sentence"), "num_labels": 2},
            "qqp": {"keys": ("question1", "question2"), "num_labels": 2},
            "rte": {"keys": ("sentence1", "sentence2"), "num_labels": 2},
            "sst2": {"keys": ("sentence", None), "num_labels": 2},
            "stsb": {"keys": ("sentence1", "sentence2"), "num_labels": 1},
        }
        sentence1_key, sentence2_key = task_to_keys[task_name]["keys"]
        num_labels = task_to_keys[task_name]["num_labels"]

        # default validation
        try:
            raw_dataset = self._load_data("nyu-mll/glue", task_name)
        except Exception as e:
            print(f"데이터셋 로드 오류: {e}")
            return None

        # all tekenizer
        # 문장 2개일때도 모두 처리
        def tokenize_function(examples):
            """datasets.map을 위한 토큰화 함수"""
            if sentence2_key:
                return self.tokenizer(examples[sentence1_key], examples[sentence2_key], padding="max_length", truncation=True, max_length=self.config.ctx_window_enc)
            else:
                return self.tokenizer(examples[sentence1_key], padding="max_length", truncation=True, max_length=self.config.ctx_window_enc)

        print("데이터셋 토큰화 중...")

        columns_to_keep = ['input_ids', 'token_type_ids', 'attention_mask', 'label', sentence1_key]
        if sentence2_key:
            columns_to_keep.append(sentence2_key)
        processed_dataset = raw_dataset.map(tokenize_function, batched=True)
        processed_dataset.set_format(type='torch', columns=columns_to_keep)

        print(f"--- 데이터 준비 완료: 총 {len(processed_dataset)}개 샘플 처리 ---")

        data_loader = DataLoader(
            processed_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        return processed_dataset, data_loader, num_labels, task_to_keys
        
    def run_dataset_inference(self, processed_dataset, data_loader, task_to_keys, output_csv_path, max_new_tokens_per_sample=50):
        """
        bert 데이터 전체를 추론하는 시간 측정
        """
        total_correct = 0
        total_samples = len(data_loader.dataset)
        processed_samples = 0
        
        # --- CSV 파일 준비 ---
        csv_file = None
        csv_writer = None
        task_name = self.data_type.name.lower()
        sentence1_key, sentence2_key = task_to_keys[task_name]["keys"] # 전역 변수 참조
        
        if output_csv_path:
            csv_headers = [sentence1_key, sentence2_key if sentence2_key else 'sentence', 'label', 'prediction']
            try:
                csv_file = open(output_csv_path, 'w', newline='', encoding='utf-8')
                csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL, escapechar='\\')
                csv_writer.writerow(csv_headers)
            except IOError as e:
                print(f"⚠️ 경고: CSV 파일을 열 수 없습니다. - {e}")

        total_gpu_time = 0.
        # start = time.time()
        sample_idx = 0

        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)

        # start_event.record()
        use_cuda_timing = (isinstance(self.device, str) and self.device.startswith("cuda") and torch.cuda.is_available())
        gpu_time_total = 0.0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating & Benchmarking"):
                try :
                    input_ids = batch['input_ids'].to(self.device)
                    input_ids = archutil.crop_data_to_ctx_window(input_ids, self.ctx_window_enc)

                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    labels = batch['label'].to(self.device)
                    batch_gpu_start = time.time()
                    # --- 모델 추론 ---
                    if use_cuda_timing:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()

                    outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                    if use_cuda_timing:
                        end_event.record(); torch.cuda.synchronize()
                        gpu_time_total += start_event.elapsed_time(end_event) / 1000.0

                    logits = outputs['logits']
                    predictions = torch.argmax(logits, dim=-1)

                    batch_gpu_end = time.time()
                    total_gpu_time += (batch_gpu_end - batch_gpu_start)
                    
                    # --- 평가: 정확도 계산 ---
                    total_correct += (predictions == labels).sum().item()
                    processed_samples += len(labels)

                    # --- CSV 저장 로직 (선택) ---
                    if output_csv_path:
                        for i in range(len(labels)):
                            original_sample = processed_dataset[sample_idx + i]
                            
                            sent1 = original_sample[sentence1_key]
                            sent2 = original_sample.get(sentence2_key, "") # sentence2가 없는 경우 대비
                            
                            csv_writer.writerow([
                                sent1,
                                sent2,
                                labels[i].item(),
                                predictions[i].item()
                            ])
                    sample_idx += len(labels)
                except torch.cuda.OutOfMemoryError:
                    print("cuda OOM")
                    torch.cuda.empty_cache()
                    if csv_file:
                        csv_file.close()
                    return None
                except Exception as e:
                    print(f"❌ 추론 실행 중 예상치 못한 오류 발생: {e}")
                    return None
                
            
        # --- 3. 전체 루프 종료 후 시간 측정 완료 ---
        # end = time.time()
        # total_time_v2 = end - start
        end_event.record()
        torch.cuda.synchronize()

        total_time = start_event.elapsed_time(end_event) / 1000.0
        if csv_file:
            csv_file.close()

        # --- 4. 최종 결과 집계 ---
        accuracy = total_correct / processed_samples if processed_samples > 0 else 0
        samples_per_sec = processed_samples / total_time if total_time > 0 else 0

        results = {
            "dataset_name": self.data_type.name,
            "processed_samples": processed_samples,
            "total_samples": total_samples,
            "accuracy": f"{accuracy:.2%}",
            "total_inference_time_sec": round(gpu_time_total, 2),
            "samples_per_sec": round(samples_per_sec, 2),
            "total_gpu_time_sec": round(total_gpu_time, 2) # NOTE: Inference time that Brain AI cares!
        }
        
        # --- 결과 출력 및 반환 ---
        print("\n[최종 평가 및 벤치마크 결과]:")
        for key, value in results.items():
            print(f"  - {key}: {value}")
            
        return results

def run(archtype, datatype, config, cuda_idx, seed):
    tmplutil.set_seed(seed)

    # work
    work_runner = BertWork(arch_type=archtype, data_type=datatype, config=config, device=f"cuda:{cuda_idx}")
    
    # data & model
    prepared_data, data_loader, num_labels, task_to_keys = work_runner._prepare_data()
    if data_loader is None:
        print("데이터 준비에 실패하여 프로그램을 종료합니다.")
        return
    work_runner._setup_model(num_labels)
    if work_runner.model is None:
        print("모델 로딩에 실패하여 프로그램을 종료합니다.")
        return

    # inference
    results = work_runner.run_dataset_inference(prepared_data, data_loader,task_to_keys,output_csv_path='./res_bert_inference.csv')
    if results is None:
        print("\n추론 과정에서 오류가 발생하여 벤치마크를 중단했습니다.")
    else:
        print("\n[최종 측정 결과]:", results)

    # Memory management: make sure to delete references to the objects
    del work_runner, prepared_data, data_loader, num_labels, task_to_keys, results
    
if __name__ == "__main__":
    datatype = DataTypeBert.MNLI
    config = Config(
            vocab_size=30522,
            batch_size=32,
            ctx_window_enc=512,
            ctx_window_dec=512,
            d_emb=128,
            d_q=64,
            d_k=64,
            d_v=64,
            d_ff=512,
            n_heads_enc=2,
            n_heads_dec_sa=2,
            n_heads_dec_ca=2,
            n_layers_enc=2,
            n_layers_dec=2,
            dropout_rate_enc=0.1,
            dropout_rate_dec=0.1
        )
    
    run(archtype=ArchType.BERT, datatype=datatype, config=config, cuda_idx=0, seed=2025)

