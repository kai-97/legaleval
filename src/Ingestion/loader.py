from bs4 import BeautifulSoup
import pdfplumber

folder_prefix = "data/raw/"

def load_html(file_name: str) -> dict:

    SELECTORS = [
        ("div", {"id": "tab_default_1"}),
        ("div", {"class": "field--name-body"}),
        ("main", {"role": "main"})
    ]

    with open(folder_prefix+file_name, encoding="utf-8") as file:
        soup = BeautifulSoup(file.read(), "lxml")
    
    for tag, attributes in SELECTORS:
        present = soup.find(tag, attributes)
        if present:
            body = present.get_text(separator="\n", strip=True)
            if len(body) > 200:
                break
        else:
            print(f"{attributes} not present in {file_name}")
    
    if not body:
        raise ValueError(f"No usable content container in {file_name}")
    
    heading = soup.find("h1").get_text(strip=True)

    file_dict = {"file_name": file_name,
                 "heading": heading,
                 "body": body}
    #print(f"{file_name} Processed Successfully")
    return file_dict


def load_pdf(file_name: str) -> dict:
    with pdfplumber.open(folder_prefix+file_name) as file:
        text = "\n".join(page.extract_text() for page in file.pages)

    file_dict = {"file_name": file_name,
                 "heading": None,
                 "body": text}
    #print(f"{file_name} Processed Successfully")
    return file_dict

if __name__ == "__main__":
    file = load_pdf("mcdonnell_douglas_v_green.pdf")
    print(file)