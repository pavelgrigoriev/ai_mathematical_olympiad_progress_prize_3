```python
!git clone https://github.com/pavelgrigoriev/ai_mathematical_olympiad_progress_prize_3.git repo
%cd repo

!pip install -q -r requirements.txt
```

```python
!mkdir -p /kaggle/working/output_datasets

!python main.py \
    --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
    --output_dir "/kaggle/working/output_datasets" \
    --num_samples -1 \
    --batch_size 128 \
    --tp_size 1 \
    --chunk_size 1000
```
