from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="knowledgator/gliner-relex-large-v1.0", 
    local_dir=r'D:\Projects\graph_rag\models\knowledgator--gliner-relex-large-v1.0'
    )
    