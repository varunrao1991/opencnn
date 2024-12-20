#pragma once

#include <memory>
#include <vector>

template<class T>
class LinkedList : public std::enable_shared_from_this<LinkedList<T>>
{
  public:
    explicit LinkedList(std::shared_ptr<T> layer);
    virtual ~LinkedList();
    void AddInput(std::shared_ptr<LinkedList<T>> link);
    std::shared_ptr<T> GetEntity();
    const std::vector<std::shared_ptr<LinkedList<T>>> &GetOutputs();
    const std::vector<std::shared_ptr<LinkedList<T>>> &GetInputs();
    bool IsProcessed();
    void SetProcessed();
    void ResetProcessed();

  protected:
    void addOutput(std::shared_ptr<LinkedList<T>> link);

  private:
    std::shared_ptr<T> m_entity;
    bool m_isProcessed;
    std::vector<std::shared_ptr<LinkedList<T>>> m_inputLinks;
    std::vector<std::shared_ptr<LinkedList<T>>> m_outputLinks;
};

template<class T>
LinkedList<T>::LinkedList(std::shared_ptr<T> layer) : m_entity{ layer }, m_isProcessed{ false }
{
}

template<class T>
inline LinkedList<T>::~LinkedList() = default;

template<class T>
inline void LinkedList<T>::AddInput(std::shared_ptr<LinkedList<T>> link)
{
    m_inputLinks.push_back(link);
    std::shared_ptr<LinkedList<T>> sharedThis = this->shared_from_this();
    link->addOutput(sharedThis);
}

template<class T>
inline std::shared_ptr<T> LinkedList<T>::GetEntity()
{
    return m_entity;
}

template<class T>
inline const std::vector<std::shared_ptr<LinkedList<T>>> &LinkedList<T>::GetOutputs()
{
    return m_outputLinks;
}

template<class T>
inline const std::vector<std::shared_ptr<LinkedList<T>>> &LinkedList<T>::GetInputs()
{
    return m_inputLinks;
}

template<class T>
inline bool LinkedList<T>::IsProcessed()
{
    return m_isProcessed;
}

template<class T>
inline void LinkedList<T>::SetProcessed()
{
    m_isProcessed = true;
}

template<class T>
inline void LinkedList<T>::ResetProcessed()
{
    m_isProcessed = false;
}

template<class T>
inline void LinkedList<T>::addOutput(std::shared_ptr<LinkedList<T>> link)
{
    m_outputLinks.push_back(link);
}
