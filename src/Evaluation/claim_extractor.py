from dataclasses import dataclass

import nltk

for pkg in ("punkt_tab", "punkt"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)

@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str


def extract_claims(answer: str) -> list[Claim]:
    """Split answer text into Claim objects with stable ids."""
    sentences = nltk.sent_tokenize(answer)
    result = []
    for id, sentence in enumerate(sentences):
        if len(sentence.split(" ")) < 4:
            continue
        result.append(Claim(f"c{id:02d}", sentence))

    return result


if __name__ == "__main__":
    test_answer = ("In Young v. United Parcel Service, the Supreme Court "
    "addressed pregnancy discrimination claims. The court "
    "held that an employer's neutral policy can still "
    "violate the Pregnancy Discrimination Act.")
    
    print(extract_claims(test_answer))