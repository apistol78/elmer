/*
 * TRAKTOR
 * Copyright (c) 2026 Anders Pistol.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "Core/Containers/AlignedVector.h"
#include "Core/Object.h"
#include "Core/Ref.h"
#include "Core/Thread/Semaphore.h"
#include "Core/Thread/Signal.h"
#include "Llm/ChatTemplate.h"
#include "Llm/Sampler.h"
#include "Llm/Trace.h"

#include <string>

// import/export mechanism.
#undef T_DLLCLASS
#if defined(T_LLM_EXPORT)
#	define T_DLLCLASS T_DLLEXPORT
#else
#	define T_DLLCLASS T_DLLIMPORT
#endif

namespace traktor
{

class Thread;

}

namespace traktor::llm
{

class Context;
class Model;

/*! What the generator is doing right now. */
enum class GeneratorState
{
	Idle,
	Prompt,		//!< Reading the conversation into the key/value cache.
	Generating, //!< Producing the reply.
	Waiting,	//!< Holding a distribution for the user to pick the next token from.
	Finished,
	Cancelled,
	Failed
};

/*! Runs generation on a thread of its own and buffers the reply.
 * \ingroup Llm
 *
 * Nothing here calls back into the caller's thread. Output accumulates and
 * is collected with flushText, which lets a user interface stream the reply
 * from a timer without any cross thread call into its widgets.
 *
 * Generation must not run on a job worker, since the forward pass forks work
 * onto the job manager; the private thread keeps that separation.
 */
class T_DLLCLASS Generator : public Object
{
	T_RTTI_CLASS;

public:
	virtual ~Generator();

	bool create(const Model* model, int32_t contextLength);

	void destroy();

	/*! Start replying to \a messages.
	 *
	 * Returns false if a reply is already being generated. Turns that no
	 * longer fit the context are dropped from the front, oldest first, with
	 * any system turn kept.
	 */
	bool begin(const AlignedVector< ChatMessage >& messages, int32_t maximumTokens);

	/*! Ask generation to stop; it ends after the token in flight. */
	void cancel();

	GeneratorState getState() const;

	bool isBusy() const;

	/*! Take the text produced since the last call.
	 *
	 * Never splits a character: bytes of an unfinished one are held back
	 * until the rest arrives.
	 */
	std::string flushText();

	/*! Reason the last run failed, if it did. */
	std::wstring getMessage() const;

	int32_t getPromptTokenCount() const;

	int32_t getGeneratedTokenCount() const;

	/*! Tokens per second while reading the prompt; zero until measured. */
	double getPromptRate() const;

	/*! Tokens per second while generating; zero until measured. */
	double getGenerateRate() const;

	/*! Positions used of the context, and how many there are. */
	void getContextUsage(int32_t& outUsed, int32_t& outAvailable) const;

	void setSamplerSettings(const SamplerSettings& settings);

	SamplerSettings getSamplerSettings() const;

	/*! Forget the conversation, so the next prompt starts from nothing. */
	void resetConversation();

	/*! Describe every step of generation as it happens.
	 *
	 * Off by default. On, each prompt token read, each token drawn and the
	 * distribution it was drawn from is recorded as a TraceEvent; collect
	 * them with flushTrace, or they accumulate until the generator is
	 * destroyed. Takes effect from the next begin.
	 */
	void setTraceEnabled(bool enable);

	bool getTraceEnabled() const;

	/*! Take the events recorded since the last call, oldest first. */
	void flushTrace(AlignedVector< TraceEvent >& outEvents);

	/*! Let the user pick every token of the reply.
	 *
	 * On, each step records the distribution as a ChoiceOffered event, with
	 * the sampler's own draw as the suggestion, and waits in the Waiting
	 * state until chooseToken answers or cancel ends the run. Implies
	 * tracing. Takes effect from the next step.
	 */
	void setInteractive(bool enable);

	bool getInteractive() const;

	/*! Answer a pending choice with \a token; negative takes the suggestion.
	 *
	 * \return False if no choice is pending.
	 */
	bool chooseToken(int32_t token);

private:
	Ref< const Model > m_model;
	Ref< Context > m_context;
	Ref< Sampler > m_sampler;
	Thread* m_thread = nullptr;
	Signal m_request;
	mutable Semaphore m_lock;

	AlignedVector< ChatMessage > m_messages;
	AlignedVector< int32_t > m_evaluated; //!< Tokens already in the cache.
	int32_t m_maximumTokens = 0;
	GeneratorState m_state = GeneratorState::Idle;
	std::wstring m_message;
	std::string m_text;
	std::string m_partial; //!< Trailing bytes of an unfinished character.
	int32_t m_promptTokenCount = 0;
	int32_t m_generatedTokenCount = 0;
	double m_promptRate = 0.0;
	double m_generateRate = 0.0;
	bool m_cancel = false;
	bool m_traceEnabled = false;
	AlignedVector< TraceEvent > m_trace;
	bool m_interactive = false;
	Signal m_choice;
	int32_t m_chosenToken = -1;

	void threadGenerate();

	void generate();

	/*! Tokenize \a messages, dropping old turns until the result fits.
	 *
	 * \param outDroppedTurns How many turns had to go.
	 * \param outPromptBytes Length of the formatted prompt that was kept.
	 */
	bool buildPrompt(const AlignedVector< ChatMessage >& messages, int32_t maximumTokens, AlignedVector< int32_t >& outTokens, int32_t& outDroppedTurns, int32_t& outPromptBytes);

	void appendText(const std::string& text);

	void setState(GeneratorState state, const std::wstring& message);

	void pushTrace(const TraceEvent& event);
};

}
