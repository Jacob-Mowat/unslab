from data.loader import RAWLoader

def main():
    fronts_loader = RAWLoader("/Users/jacob/documents/unprocessed_slabs/CharizardEX/front/")
    fronts_loader.load_raw_images()
    fronts_loader.post_process_raw_images()

    images = fronts_loader.get_images()
    print(images)


if __name__ == "__main__":
    main()
