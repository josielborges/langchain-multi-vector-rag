# Retrieve using multiple vectors per document

It can often be useful to store multiple vectors per document. There are multiple use cases where this is beneficial. For example, we can embed multiple chunks of a document and associate those embeddings with the parent document, allowing retriever hits on the chunks to return the larger document.

LangChain implements a base MultiVectorRetriever, which simplifies this process. Much of the complexity lies in how to create the multiple vectors per document. This notebook covers some of the common ways to create those vectors and use the MultiVectorRetriever.

The methods to create multiple vectors per document include:

Smaller chunks: split a document into smaller chunks, and embed those (this is ParentDocumentRetriever).
Summary: create a summary for each document, embed that along with (or instead of) the document.
Hypothetical questions: create hypothetical questions that each document would be appropriate to answer, embed those along with (or instead of) the document.
Note that this also enables another method of adding embeddings - manually. This is useful because you can explicitly add questions or queries that should lead to a document being recovered, giving you more control.

References

- [Langchain docs](https://python.langchain.com/docs/how_to/multi_vector/)
- [Eric Vaillancourt - Youtube Video](https://www.youtube.com/watch?v=weQpcue-0G4)
- [Eric Vaillancourt - Github](https://github.com/ericvaillancourt/LangChain_persistant_multi_vector)
- [Eric Vaillancourt - Medium](https://medium.com/@eric_vaillancourt/enough-with-prototyping-time-for-persistent-multi-vector-storage-with-postgresql-in-langchain-8e678738e80d)
