# 基於內容之機器推薦中文文章系統(Content-based Machine Recommender System in Chinese article)

## 問題定義
現代的小說推薦系統以及詩詞推薦系統都是利用collaborative-filter來做推薦，或是簡單的運用類型來做基礎的推薦，致使使用者無法得知自己想要閱讀的文章。此系統解決只運用content-based來推薦傳統中華文學，如詩詞和小說，可在不受別人喜好影響下推薦文章，並且在推薦的同時，可以快速了解此篇文章的基礎資訊，提供人以內容為導向閱讀自己喜歡的文章。
## 功能介紹
* 推薦文章：兩種推薦方法，一種依循關鍵字和作者寫作風格推薦詩詞和小說，第二種為依尋關鍵字回傳詩詞和小說類別。
* 產生文章：輸入狀況產生詩詞。
* 文章概覽：可查詢和瀏覽詩詞的作者、風格、題目、年代、部份內容、ＦＤＡ產物和文字內容難度。
* 圖片支援: 在查詢階段，透過關鍵字提取找到相對應的圖片做支援，並加入文字雲讓人可以快速了解詩詞或小說內容。
* 建立自己的Profile，來推薦不同類型的文章。
* 與詩詞對話

## 推薦方法
* First:
	* input: word
	* outputs: author(tag), article(tag)
	* ways: BERT(2 models), LDA
* Second Search:
	* input: words or author
	* outputs: the article and similar words
	* ways: same words
* Third:
	* input: Selected button(tag)
	* outputs: author(tag), article(tag)
	* ways: Selected LDA topic
* Forth:
	* input: Selected picture, plus OCR 
	* outputs: article
	* ways: OCR transform to text

## Other function
* 與文章對話(novel): 
	* input: words
	* outputs: some words
	* ways: language model
* 探討古人:
	* input: author
	* outputs: search result
	* ways: text to saved the text
* Quick Look of article
	* input: title
	* outputs: words cloud
	* ways: wordclouds
* Generate poem(poem):
	* input: situation
	* outputs: poem
	* ways: language model
