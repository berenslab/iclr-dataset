from openreview.api import OpenReviewClient
from pathlib import Path
import logging
import pyarrow
import pyarrow.parquet as pq
from collections import OrderedDict
import json, os
from tqdm import tqdm

logger = logging.getLogger(__name__)


def find_thread_id(reply_id, submission_id, replyto_mappings):
    replyto = replyto_mappings.get(reply_id)
    if replyto is None:
        return None
    if replyto == submission_id:
        return reply_id
    return find_thread_id(replyto, submission_id, replyto_mappings)


def content_dict_to_text(content_dict):
    main_keys = [
        "title",
        "summary",
        "strengths",
        "weaknesses",
        "questions",
        "comment",
        "metareview",
    ]
    text = ""
    for k in main_keys:
        data = content_dict.get(k)
        if data is None:
            continue
        text += f"\n{k}:\n{data['value']}\n"
    return text.strip()


data_folder = Path("./data/")
data_folder.mkdir(parents=True, exist_ok=True)
raw_dataset_path = data_folder / "iclr_2025_raw.parquet"
massaged_data_path = data_folder / "iclr_2025_dataset.json"
reviews_folder = data_folder / "reviews"
reviews_folder.mkdir(parents=True, exist_ok=True)


client = OpenReviewClient(baseurl="https://api2.openreview.net")


if raw_dataset_path.exists():
    logger.info(f"Loading the raw ICLR 2025 dataset from {raw_dataset_path}")
    table = pq.read_table(raw_dataset_path)
    all_submissions = table.to_pylist()
else:
    logger.info("Fetching all submissions from OpenReview API...")
    # Fetch notes from the API. This returns a list of openreview.Note objects.
    all_submissions_notes = client.get_all_notes(
        invitation="ICLR.cc/2025/Conference/-/Submission", details="replies"
    )

    # Convert the list of Note objects into a list of dictionaries
    all_submissions = [
        {"metadata": note.to_json(), "replies": note.details["replies"]}
        for note in all_submissions_notes
    ]

    if all_submissions:
        # Create a PyArrow Table from the list of dictionaries
        table = pyarrow.Table.from_pylist(all_submissions)
        pq.write_table(table, raw_dataset_path)
        logger.info(f"Saved the raw ICLR 2025 dataset at {raw_dataset_path}")
    else:
        logger.warning(
            "No submissions found; an empty dataset file will not be created."
        )
if massaged_data_path.exists():
    massaged_data = json.load(open(massaged_data_path, encoding="utf-8"))
else:
    massaged_data = dict()

for sub in tqdm(all_submissions, desc="Processing ICLR submissions"):
    metadata = sub.get("metadata")
    sub_id = metadata["id"]
    if os.path.exists(reviews_folder / f"{sub_id}.json"):
        continue
    keep_keys = ["cdate", "content", "forum"]
    metadata.update({k: v for k, v in metadata.items()})
    replies = sub.get("replies", [])

    replyto_mappings = {r["id"]: r["replyto"] for r in replies}
    threads = dict()
    for reply in replies:
        reply_id = reply["id"]
        writer = reply["writers"][-1].split("/")[-1]
        thread_id = find_thread_id(reply_id, sub_id, replyto_mappings)
        if thread_id is None:
            continue
        writer_type = reply["signatures"][0].split("/")[-1].lower()
        thread_type = reply["invitations"][0].split("/")[-1].lower()

        if thread_id == reply_id:  # it's a new thread
            # find what type of thread this is: review/metareview/comment (author)/decision
            if thread_type == "decision":
                # update paper metadata rather than creating a new thread
                metadata["content"]["decision"] = reply["content"]["decision"]
                continue
            threads[reply_id] = {
                "cdate": reply["cdate"],
                "type": thread_type,
                "content": [],
            }
        if thread_id not in threads:
            continue

        # add this reply object to the appropriate thread
        threads[thread_id]["content"].append(
            {
                "cdate": reply["cdate"],
                "writer": writer_type,
                "content": content_dict_to_text(reply["content"]),
            }
        )
    # sort all threads by their creation date (cdate)
    threads = dict(sorted(threads.items(), key=lambda item: item[1]["cdate"]))
    # sort all content list (objects inside those threads) by their creation date
    for thread_id, _ in threads.items():
        content = sorted(threads[thread_id]["content"], key=lambda o: o["cdate"])

        threads[thread_id]["content"] = content
    complete_discussion = ""

    # loop over threads
    for thread_id, thread_data in threads.items():
        thread_content_blocks = []

        # build content for each object in the thread
        for obj in thread_data["content"]:
            thread_object = (
                "\n<thread_object>\n"
                f"creator: {obj['writer']}\n\n"
                "<content>\n"
                f"{obj['content']}\n"
                "</content>\n"
                "</thread_object>\n"
            )
            thread_content_blocks.append(thread_object)

        # Join all thread objects and assemble the thread block
        thread_block = (
            "<thread>\n"
            f"creation date:\n{thread_data['cdate']}\n\n"
            f"thread_type:\n{thread_data['type']}\n"
            f"{''.join(thread_content_blocks)}\n"
            "</thread>\n"
        )

        complete_discussion += thread_block

    massaged_data.update(
        {sub_id: {"metadata": metadata, "discussion": complete_discussion}}
    )
    # save individual submission thread inside the thread folder
    with open(reviews_folder / f"{sub_id}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "discussion_flat": complete_discussion,
                "threads": threads,
            },
            f,
            indent=2,
        )
# update the big json dict
with open(massaged_data_path, "w") as full_data:
    json.dump(massaged_data, full_data, indent=2)
