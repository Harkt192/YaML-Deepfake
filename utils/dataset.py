import kagglehub

kaggle_api = "KGAT_8eb0727ee6d566add6db98054cc75b1f"
kagglehub.login()

kagglehub.competition_download(
    "ml-intensive-yandex-academy-spring-2026",
    output_dir="./data"
)