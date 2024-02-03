if parameter == "history":
    response = gpt_Bot.reset_history(thread_ts)
    say(f"`{response}`", thread_ts=thread_ts)


def reset_history(self, thread_id):
    if not thread_id:
        return "!reset history can only be run inside of a thread."

    self.conversations[thread_id]["messages"] = [self.SYSTEM_PROMPT]
    self.usage = {}  # Figure out what to do with usage stats in conversations
    self.conversations[thread_id]["processing"] = False

    return "Rebooting. Beep Beep Boop. My memory has been wiped!"
