from dspy.utils import download
def main():
    # download_file()
    print("Downloading file...")
    download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl")


if __name__ == "__main__":
    main()
