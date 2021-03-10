import numpy as np


class Library:

    def __init__(self, id, book_ids, signup_days, max_books_scanned_per_day):
        self.id = id
        self.book_ids = np.array(book_ids)
        self.signup_days = signup_days
        self.max_books_scanned_per_day = max_books_scanned_per_day
        self.is_signed = False

    def scan_books(self):  # TODO: scan only books not scanned already
        if self.is_signed:
            self.book_ids = self.book_ids[
                            self.max_books_scanned_per_day:]  # TODO: env chooses which books to scan also?
            return self.max_books_scanned_per_day  # TODO: different scores for each book
        else:
            return 0
