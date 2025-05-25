import requests
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🎨 Theme Colors
BG_COLOR = "#f9f9fc"         # light background
ACCENT_COLOR = "#d0bfff"     # pastel purple
TEXT_COLOR = "#3b2e5a"       # deep violet
FONT = ("Helvetica", 11)
TITLE_FONT = ("Helvetica", 14, "bold")

BASE_URL = "https://api.jikan.moe/v4"

def search_anime(query, limit=10):
    url = f"{BASE_URL}/anime"
    params = {"q": query, "limit": limit}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return [
        {
            "title": anime["title"],
            "mal_id": anime["mal_id"],
            "synopsis": anime["synopsis"] or ""
        }
        for anime in resp.json().get("data", [])
        if anime.get("synopsis")
    ]

def recommend_anime(base_title, anime_list, top_n=3):
    titles = [anime["title"] for anime in anime_list]
    synopses = [anime["synopsis"] for anime in anime_list]

    try:
        base_index = titles.index(base_title)
    except ValueError:
        return f"'{base_title}' not found in search results."

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(synopses)
    cosine_sim = cosine_similarity(tfidf_matrix[base_index], tfidf_matrix).flatten()

    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    recommendations = [{"title": titles[i], "score": round(cosine_sim[i], 3)} for i in similar_indices]

    return recommendations

def on_search():
    query = entry.get().strip()
    if not query:
        messagebox.showwarning("Input Error", "Please enter an anime name.")
        return

    try:
        anime_data = search_anime(query, limit=20)
        base_title = query

        results = recommend_anime(base_title, anime_data, top_n=5)

        result_box.delete(0, tk.END)

        if isinstance(results, str):
            result_box.insert(tk.END, results)
        else:
            for rec in results:
                result_box.insert(tk.END, f"{rec['title']} (Similarity: {rec['score']})")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI Setup
root = tk.Tk()
root.title("🌸 Anime Recommender")
root.configure(bg=BG_COLOR)
root.resizable(False, False)

frame = tk.Frame(root, bg=BG_COLOR, padx=20, pady=20)
frame.pack()

title_label = tk.Label(frame, text="✨ What's your favorite anime?", font=TITLE_FONT, bg=BG_COLOR, fg=TEXT_COLOR)
title_label.pack(anchor="w")

entry = tk.Entry(frame, width=40, font=FONT, bg="white", fg=TEXT_COLOR, highlightthickness=1, highlightbackground=ACCENT_COLOR)
entry.pack(pady=6)

search_button = tk.Button(frame, text="🌟 Get Recommendations", command=on_search,
                          bg=ACCENT_COLOR, fg="black", font=FONT, bd=0, padx=10, pady=5,
                          activebackground="#c0a7f9", activeforeground="white", relief="flat")
search_button.pack(pady=8)

list_frame = tk.Frame(frame, bg=BG_COLOR)
list_frame.pack()

scrollbar = tk.Scrollbar(list_frame)
result_box = tk.Listbox(list_frame, width=50, height=10, font=FONT, yscrollcommand=scrollbar.set,
                        bg="white", fg=TEXT_COLOR, highlightthickness=1, highlightbackground=ACCENT_COLOR,
                        selectbackground=ACCENT_COLOR)
scrollbar.config(command=result_box.yview)

result_box.pack(side=tk.LEFT, fill=tk.BOTH)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

root.mainloop()
