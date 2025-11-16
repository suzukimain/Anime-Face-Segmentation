call .venv\Scripts\activate

python predict.py --src_path Hoshino.webp  --save_dir test_img --model_path model/checkpoint.pth  > result.log 2>&1
