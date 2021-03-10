# hashcode2020qualification_RL
## Road map
1. Successfully run a known example in the framework.
1. Finetune a known example in the framework.
   1. Get env of known example.
   1. Implement/copy training code (agent).
1. Parse data.
1. Implement gym env.
   1. First just pick which library to sign up at each point (all books are unique and same score).
   1. Then also implement library **and** book selection with unique books (each needs to be scanned only once).
1. Train.
1. Win.

## Notes
1. First win a **specific** setting.
1. Check what how to reward:
    1. Only at last day.
    1. Each day.
    1. Every n days.
1. Discount factor should be high (we don't care when the score is achieved as long as it's in D days).


## Version 1
* Each `library` has a unique set of `books`.
* `reward` is given at the end of the episode.
* Each `library` takes 1 day to sign up.
