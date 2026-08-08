from rag_v2.config import Stage1ChunkParams
from rag_v2.ingestion.chunker import build_chunk_stats, build_chunks, make_embedding_text
from rag_v2.ingestion.markdown_parser import MarkdownBlock
from rag_v2.ingestion.token_counter import RegexTokenCounter, split_by_token_limit


def block(text, section=None, source="doc.md", indexable=True):
    return MarkdownBlock(
        source_file=source,
        section_path=section or ["doc", "section"],
        text=text,
        block_type="paragraph",
        is_indexable=indexable,
        start_line=1,
        end_line=1,
    )


def test_regex_token_counter_basic_behavior():
    counter = RegexTokenCounter()
    assert counter.tokenize("Climate change 2024") == ["Climate", "change", "2024"]
    assert counter.count("气象灾害IV级") >= 6


def test_split_by_token_limit_respects_limit():
    counter = RegexTokenCounter()
    pieces = split_by_token_limit("灾" * 25, counter, 10)
    assert len(pieces) == 3
    assert all(counter.count(piece) <= 10 for piece in pieces)


def test_build_chunks_hard_cuts_oversized_single_block():
    params = Stage1ChunkParams(target_tokens=50, soft_max_tokens=60, hard_max_tokens=80, overlap_tokens=10, min_tokens=5)
    counter = RegexTokenCounter()
    chunks = build_chunks([block("灾" * 240)], params=params, token_counter=counter)
    assert len(chunks) > 1
    assert all(chunk.token_count <= params.hard_max_tokens for chunk in chunks)
    assert all(chunk.source_file == "doc.md" for chunk in chunks)


def test_overlap_is_added_between_adjacent_chunks():
    params = Stage1ChunkParams(target_tokens=45, soft_max_tokens=55, hard_max_tokens=75, overlap_tokens=8, min_tokens=5)
    counter = RegexTokenCounter()
    text = "。".join(["灾" * 20 for _ in range(8)]) + "。"
    chunks = build_chunks([block(text)], params=params, token_counter=counter)
    assert len(chunks) >= 2
    previous_tokens = counter.tokenize(chunks[0].content)[-params.overlap_tokens :]
    second_tokens = counter.tokenize(chunks[1].content)[: params.overlap_tokens]
    assert second_tokens == previous_tokens
    assert all(chunk.token_count <= params.hard_max_tokens for chunk in chunks)


def test_overlap_does_not_cross_section_boundary():
    params = Stage1ChunkParams(target_tokens=30, soft_max_tokens=40, hard_max_tokens=70, overlap_tokens=8, min_tokens=5)
    chunks = build_chunks(
        [
            block("甲" * 60, section=["doc", "A"]),
            block("乙" * 20, section=["doc", "B"]),
        ],
        params=params,
        token_counter=RegexTokenCounter(),
    )
    section_b = [chunk for chunk in chunks if chunk.section_path == ["doc", "B"]][0]
    assert "甲" not in section_b.content
    assert section_b.content.startswith("乙")


def test_non_indexable_blocks_are_ignored():
    chunks = build_chunks([block("目录 1", indexable=False), block("真正正文。")], token_counter=RegexTokenCounter())
    assert len(chunks) == 1
    assert chunks[0].content == "真正正文。"


def test_chunk_ids_and_indexes_are_per_source_file():
    params = Stage1ChunkParams(target_tokens=30, soft_max_tokens=40, hard_max_tokens=70, overlap_tokens=5, min_tokens=5)
    chunks = build_chunks(
        [
            block("甲" * 60, source="a.md", section=["a"]),
            block("乙" * 60, source="b.md", section=["b"]),
        ],
        params=params,
        token_counter=RegexTokenCounter(),
    )
    a_indexes = [chunk.chunk_index for chunk in chunks if chunk.source_file == "a.md"]
    b_indexes = [chunk.chunk_index for chunk in chunks if chunk.source_file == "b.md"]
    assert a_indexes == list(range(len(a_indexes)))
    assert b_indexes == list(range(len(b_indexes)))
    assert all(chunk.chunk_id == f"{chunk.source_file}::{chunk.chunk_index}" for chunk in chunks)


def test_embedding_text_contains_doc_section_and_content_but_content_is_clean():
    chunks = build_chunks([block("气象部门加强监测预报。", source="深圳市气象灾害应急预案.md", section=["深圳市气象灾害应急预案", "4 应急响应"])])
    chunk = chunks[0]
    assert "[文档] 深圳市气象灾害应急预案" in chunk.embedding_text
    assert "[章节] 深圳市气象灾害应急预案 > 4 应急响应" in chunk.embedding_text
    assert "[正文] 气象部门加强监测预报。" in chunk.embedding_text
    assert chunk.content == "气象部门加强监测预报。"
    assert "[文档]" not in chunk.content


def test_make_embedding_text_fallback_path():
    text = make_embedding_text("a.md", [], "正文")
    assert "[文档] a" in text
    assert "[章节] a" in text


def test_build_chunk_stats_summary():
    blocks = [block("正文一。"), block("目录", indexable=False)]
    chunks = build_chunks(blocks)
    stats = build_chunk_stats("doc.md", blocks, chunks)
    assert stats.input_blocks == 2
    assert stats.indexable_blocks == 1
    assert stats.output_chunks == 1
    assert stats.max_tokens == chunks[0].token_count
    assert stats.missing_section_path_chunks == 0
    assert stats.to_dict()["source_file"] == "doc.md"
