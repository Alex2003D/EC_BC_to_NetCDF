def unix_id_from_seconds(timestamp_seconds):
    """Create measurement ID by replacing first digit with '2'.
    """
    s = str(int(timestamp_seconds))
    if len(s) > 1:
        return int("2" + s[1:])
    return int("2" + s)


