# Summarize the content of all the websites posted to the 'general' channel and post
# the summary to the 'random' channel. Single self-contained fragment.
messages = read_channel_messages("general")

all_messages = "\n".join([f"{m.sender}: {m.body}" for m in messages])

class UrlList(BaseModel):
    urls: list[str]

url_list = query_ai_assistant(
    f"The following are Slack messages from the 'general' channel. Extract every website URL that is posted in these messages, exactly as written:\n{all_messages}",
    output_schema=UrlList,
)

pages = [f"Content of {url}:\n{get_webpage(url)}" for url in url_list.urls]
all_pages = "\n\n".join(pages)

class Summary(BaseModel):
    summary: str

result = query_ai_assistant(
    f"Write a concise summary of the content of the following webpages, which were posted to the 'general' Slack channel:\n{all_pages}",
    output_schema=Summary,
)

send_channel_message("random", result.summary)
print(f"Posted summary to 'random' channel: {result.summary}")
