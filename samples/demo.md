# RAG 检索学习笔记

RAG 的完整流程通常包括解析文档、切块、生成 embedding、写入向量库、查询召回、rerank，最后才是生成答案。

## Embedding

Embedding 会把文本转换成向量。语义相近的文本，在向量空间里的方向往往更接近，所以可以用 Cosine Similarity 衡量相似度。

## Chunking

Chunking 不是越小越好，也不是越大越好。太小会丢上下文，太大会让召回结果包含太多无关内容。

## Metadata

Metadata 用来保存来源、页码、标题、权限、标签和时间。检索时可以用 metadata 做过滤，例如只搜某个项目或某类文档。

## Vector DB

Qdrant、pgvector、Milvus、Pinecone 和 Weaviate 都可以存储向量。本项目选择 Qdrant，因为它支持本地模式，适合学习。

## Top-k 和 Rerank

Top-k 决定初步召回多少条。Rerank 会在初步召回后重新排序，让最终交给大模型的片段更准确。
