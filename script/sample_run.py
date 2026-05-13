from retrieve import retrieve

if __name__ == "__main__":

    query = "a dog playing in park"

    results = retrieve(query, top_k=5)

    print("\nQuery:", query)
    print("Retrieved Images:\n")

    for r in results:
        print(r)
