"""A tiny, public-domain text sample for the S1 LOCAL SMOKE TEST ONLY.

Not the production data mix -- see configs/s1.py's TODO(Jun) (slide 30:
Code 15% / Japanese 10% / Math 5% / EN-web rest, exact dataset ids need
Jun). This exists purely so scripts/build_smoke_shard.py can produce a
few-MB, git-ignored mini-shard without any network dependency beyond the
Qwen3 tokenizer (already locally HF-cached from S0) and without adding a
new `datasets` package dependency just to run a 0-CU smoke test.

All paragraphs below are pre-1928-US-publication public domain (opening
passages of well-known out-of-copyright works), used here only as
tokenizable filler text -- content is irrelevant to what's being tested
(the crux wiring + resumable checkpointing), only token diversity/volume
matters.
"""
from typing import Iterator

PARAGRAPHS = [
    # Alice's Adventures in Wonderland, ch. 1 (Lewis Carroll, 1865)
    "Alice was beginning to get very tired of sitting by her sister on the "
    "bank, and of having nothing to do: once or twice she had peeped into "
    "the book her sister was reading, but it had no pictures or "
    "conversations in it, 'and what is the use of a book,' thought Alice "
    "'without pictures or conversations?'",
    # Pride and Prejudice, ch. 1 (Jane Austen, 1813)
    "It is a truth universally acknowledged, that a single man in "
    "possession of a good fortune, must be in want of a wife. However "
    "little known the feelings or views of such a man may be on his first "
    "entering a neighbourhood, this truth is so well fixed in the minds "
    "of the surrounding families, that he is considered the rightful "
    "property of some one or other of their daughters.",
    # A Study in Scarlet, ch. 1 (Arthur Conan Doyle, 1887)
    "In the year 1878 I took my degree of Doctor of Medicine of the "
    "University of London, and proceeded to Netley to go through the "
    "course prescribed for surgeons in the army. Having completed my "
    "studies there, I was duly attached to the Fifth Northumberland "
    "Fusiliers as Assistant Surgeon.",
    # Moby-Dick, ch. 1 (Herman Melville, 1851)
    "Call me Ishmael. Some years ago, never mind how long precisely, "
    "having little or no money in my purse, and nothing particular to "
    "interest me on shore, I thought I would sail about a little and see "
    "the watery part of the world.",
    # The Declaration of Independence (1776), preamble
    "When in the Course of human events, it becomes necessary for one "
    "people to dissolve the political bands which have connected them "
    "with another, and to assume among the powers of the earth, the "
    "separate and equal station to which the Laws of Nature and of "
    "Nature's God entitle them, a decent respect to the opinions of "
    "mankind requires that they should declare the causes which impel "
    "them to the separation.",
    # Frankenstein, letter 1 (Mary Shelley, 1818)
    "You will rejoice to hear that no disaster has accompanied the "
    "commencement of an enterprise which you have regarded with such evil "
    "forebodings. I arrived here yesterday, and my first task is to "
    "assure my dear sister of my welfare and increasing confidence in the "
    "success of my undertaking.",
    # The Adventures of Tom Sawyer, ch. 1 (Mark Twain, 1876)
    "Tom! No answer. Tom! No answer. What's gone with that boy, I wonder? "
    "You TOM! No answer. The old lady pulled her spectacles down and "
    "looked over them about the room; then she put them up and looked out "
    "under them.",
    # The Time Machine, ch. 1 (H. G. Wells, 1895)
    "The Time Traveller (for so it will be convenient to speak of him) "
    "was expounding a recondite matter to us. His grey eyes shone and "
    "twinkled, and his usually pale face was flushed and animated. The "
    "fire burned brightly, and the soft radiance of the incandescent "
    "lights in the lilies of silver caught the bubbles that flashed and "
    "passed in our glasses.",
]


def repeated(n_repeats: int) -> Iterator[str]:
    """Cycles PARAGRAPHS `n_repeats` times -- enough raw text volume for a
    ~50-step, batch-2, T=128 smoke run (with headroom for the loader's
    multi-shard wraparound), without pretending to be diverse training
    data."""
    for _ in range(n_repeats):
        yield from PARAGRAPHS
