import json
from pathlib import Path
from datetime import datetime

class WordPressUploader:
    """
    Stub for WordPress Uploader.
    Saves the post content to a JSON file instead of uploading.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def upload_post(self, title: str, content: str, images: list = None):
        """
        Simulates creating a post.
        """
        post_data = {
            "title": title,
            "content": content,
            "status": "draft",
            "uploaded_at": datetime.now().isoformat(),
            "images": images or []
        }

        # Create a safe filename from title
        safe_title = "".join([c if c.isalnum() else "_" for c in title])[:50]
        filename = f"wp_upload_{safe_title}.json"

        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(post_data, f, indent=2, ensure_ascii=False)

        print(f"[Stub] 'Uploaded' post to {filepath}")
        return {"id": 12345, "link": f"http://example.com/{safe_title}"}
