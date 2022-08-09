# Import library to read urls
from urllib.parse import urlparse
from urllib.parse import parse_qs
# Import Visual
import progressbar

class Progress:
    def __init__(self):
        self.bar = self.initiate_bar()


    def calc_update_percentage(self, url, total):
        try:
            parsed_url = urlparse(url)
            current_result = int(parse_qs(parsed_url.query)["start"][0])
            
            self.update(current_result/total*100)
        except KeyError:
            raise Exception("Couldn't find current result")


    def initiate_bar(self):
        widgets = [
            progressbar.Bar('>'), ' ',
            progressbar.ETA(), ' ',
            progressbar.ReverseBar('<'),
        ]
        return progressbar.ProgressBar(widgets=widgets, max_value=100)


    def start(self):
        self.bar.start()

    def update(self, percentage):
        self.bar.update(percentage)

    def finish(self):
        self.bar.finish()