/*
 * TRAKTOR
 * Copyright (c) 2026 Anders Pistol.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "Llm/Generator.h"

#include "Core/Log/Log.h"
#include "Core/Misc/SafeDestroy.h"
#include "Core/Misc/String.h"
#include "Core/Thread/Acquire.h"
#include "Core/Thread/Thread.h"
#include "Core/Thread/ThreadManager.h"
#include "Core/Timer/Timer.h"
#include "Llm/Context.h"
#include "Llm/Model.h"
#include "Llm/Tokenizer.h"
#include "Llm/Utf8.h"
#include "Llm/Vocabulary.h"

#include <algorithm>

namespace traktor::llm
{
namespace
{

// Never let the conversation crowd the reply out of more than this share of
// the context, however many tokens the caller asked for.
const int32_t c_maximumReserveShare = 2;

/*! Text of \a token for a trace.
 *
 * Unlike the reply, a trace wants to show chat markup, since seeing
 * "<|im_start|>" go by is the point; control tokens keep their own text.
 */
std::string tracePiece(const Tokenizer* tokenizer, const Vocabulary* vocabulary, int32_t token)
{
	if (vocabulary->getTokenType(token) == TokenType::Control)
		return vocabulary->getTokenText(token);
	return tokenizer->decode(token);
}

}

T_IMPLEMENT_RTTI_CLASS(L"traktor.llm.Generator", Generator, Object)

Generator::~Generator()
{
	destroy();
}

bool Generator::create(const Model* model, int32_t contextLength)
{
	if (model == nullptr)
		return false;

	m_model = model;

	m_context = new Context();
	if (!m_context->create(model, contextLength))
		return false;

	m_sampler = new Sampler();

	m_thread = ThreadManager::getInstance().create([this]() {
		threadGenerate();
	},
		L"LLM generator");

	if (m_thread == nullptr || !m_thread->start())
	{
		log::error << L"Unable to start generator thread." << Endl;
		return false;
	}

	return true;
}

void Generator::destroy()
{
	cancel();

	if (m_thread != nullptr)
	{
		m_thread->stop();
		ThreadManager::getInstance().destroy(m_thread);
		m_thread = nullptr;
	}

	safeDestroy(m_context);
	m_sampler = nullptr;
	m_model = nullptr;
}

bool Generator::begin(const AlignedVector< ChatMessage >& messages, int32_t maximumTokens)
{
	{
		T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);

		if (m_state == GeneratorState::Prompt || m_state == GeneratorState::Generating || m_state == GeneratorState::Waiting)
			return false;

		m_messages = messages;
		m_maximumTokens = (maximumTokens > 0) ? maximumTokens : m_context->getContextLength();
		m_state = GeneratorState::Prompt;
		m_message.clear();
		m_text.clear();
		m_partial.clear();
		m_promptTokenCount = 0;
		m_generatedTokenCount = 0;
		m_promptRate = 0.0;
		m_generateRate = 0.0;
		m_cancel = false;
	}

	m_request.set();
	return true;
}

void Generator::cancel()
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	m_cancel = true;
}

GeneratorState Generator::getState() const
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	return m_state;
}

bool Generator::isBusy() const
{
	const GeneratorState state = getState();
	return state == GeneratorState::Prompt || state == GeneratorState::Generating || state == GeneratorState::Waiting;
}

std::string Generator::flushText()
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	std::string text;
	text.swap(m_text);
	return text;
}

std::wstring Generator::getMessage() const
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	return m_message;
}

int32_t Generator::getPromptTokenCount() const
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	return m_promptTokenCount;
}

int32_t Generator::getGeneratedTokenCount() const
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	return m_generatedTokenCount;
}

double Generator::getPromptRate() const
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	return m_promptRate;
}

double Generator::getGenerateRate() const
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	return m_generateRate;
}

void Generator::getContextUsage(int32_t& outUsed, int32_t& outAvailable) const
{
	outUsed = m_context ? m_context->getPosition() : 0;
	outAvailable = m_context ? m_context->getContextLength() : 0;
}

void Generator::setSamplerSettings(const SamplerSettings& settings)
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	if (m_sampler)
		m_sampler->setSettings(settings);
}

SamplerSettings Generator::getSamplerSettings() const
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	return m_sampler ? m_sampler->getSettings() : SamplerSettings();
}

void Generator::resetConversation()
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);

	// The worker owns the evaluated token list while it runs, so refuse
	// rather than clear it out from under it.
	if (m_state == GeneratorState::Prompt || m_state == GeneratorState::Generating || m_state == GeneratorState::Waiting)
		return;

	m_evaluated.clear();
	if (m_context)
		m_context->reset();
}

void Generator::setTraceEnabled(bool enable)
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	m_traceEnabled = enable;
}

bool Generator::getTraceEnabled() const
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	return m_traceEnabled;
}

void Generator::flushTrace(AlignedVector< TraceEvent >& outEvents)
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	outEvents.swap(m_trace);
}

void Generator::setInteractive(bool enable)
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	m_interactive = enable;
}

bool Generator::getInteractive() const
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	return m_interactive;
}

bool Generator::chooseToken(int32_t token)
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	if (m_state != GeneratorState::Waiting)
		return false;

	m_chosenToken = token;
	m_choice.set();
	return true;
}

void Generator::threadGenerate()
{
	while (!m_thread->stopped())
	{
		if (!m_request.wait(100))
			continue;

		m_request.reset();

		if (m_thread->stopped())
			break;

		generate();
	}
}

void Generator::generate()
{
	AlignedVector< ChatMessage > messages;
	int32_t maximumTokens;
	bool trace;
	{
		T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
		messages = m_messages;
		maximumTokens = m_maximumTokens;

		// A choice cannot be offered without the distribution to show, so
		// interactive mode records whether asked to or not.
		trace = m_traceEnabled || m_interactive;
		m_choice.reset();
	}

	// Read afresh at every step, so the mode can change mid reply.
	const auto isInteractive = [&]() {
		T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
		return m_interactive;
	};

	Timer timer;
	double promptSeconds = 0.0;
	int32_t generated = 0;
	bool generating = false;

	// Every way out of here ends the run the same way: the state for the
	// caller, and for a trace, the closing event with the totals.
	const auto finish = [&](GeneratorState state, const std::wstring& message, const std::wstring& outcome) {
		setState(state, message);

		if (!trace)
			return;

		TraceEvent event;
		event.kind = TraceEvent::Kind::Finished;
		event.outcome = outcome;
		event.promptSeconds = promptSeconds;
		event.generateSeconds = generating ? timer.getElapsedTime() : 0.0;
		event.generatedTokenCount = generated;
		{
			T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
			event.promptTokenCount = m_promptTokenCount;
		}
		pushTrace(event);
	};

	const auto isCancelled = [&]() {
		T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
		return m_cancel;
	};

	AlignedVector< int32_t > tokens;
	int32_t droppedTurns = 0;
	int32_t promptBytes = 0;
	if (!buildPrompt(messages, maximumTokens, tokens, droppedTurns, promptBytes))
	{
		finish(GeneratorState::Failed, L"Conversation does not fit in the context.", L"Failed: the conversation does not fit in the context.");
		return;
	}

	if (tokens.empty())
	{
		finish(GeneratorState::Failed, L"Prompt is empty.", L"Failed: the prompt is empty.");
		return;
	}

	const Tokenizer* tokenizer = m_model->getTokenizer();
	const Vocabulary* vocabulary = m_model->getVocabulary();

	// Reuse whatever prefix of the cache the new prompt still agrees with.
	// Every turn repeats the whole conversation, so this is usually all but
	// the last exchange.
	size_t shared = 0;
	while (shared < m_evaluated.size() && shared + 1 < tokens.size() && m_evaluated[shared] == tokens[shared])
		++shared;

	m_context->rewind((int32_t)shared);
	m_evaluated.resize(shared);

	if (trace)
	{
		TraceEvent event;
		event.kind = TraceEvent::Kind::PromptBuilt;
		event.templateName = m_model->getChatTemplate()->getKindName();
		event.promptBytes = promptBytes;
		event.sharedTokens = (int32_t)shared;
		event.droppedTurns = droppedTurns;
		event.promptTokens = tokens;
		pushTrace(event);
	}

	timer.reset();

	setState(GeneratorState::Prompt, L"");

	for (size_t i = shared; i < tokens.size(); ++i)
	{
		if (isCancelled())
		{
			finish(GeneratorState::Cancelled, L"", L"Stopped while reading the prompt.");
			return;
		}

		{
			T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
			m_promptTokenCount = (int32_t)(i - shared + 1);
		}

		// Only the final prompt token needs logits; the rest are read purely
		// to fill the cache, and skipping the output projection for them is
		// the cheapest speedup available here.
		const bool last = (i + 1 == tokens.size());

		ForwardTrace forward;
		if (!m_context->evaluate(tokens[i], last, trace ? &forward : nullptr))
		{
			finish(GeneratorState::Failed, L"Context is full.", L"Failed: the context filled up while reading the prompt.");
			return;
		}

		m_evaluated.push_back(tokens[i]);

		if (trace)
		{
			TraceEvent event;
			event.kind = TraceEvent::Kind::PromptToken;
			event.token = tokens[i];
			event.position = forward.position;
			event.piece = tracePiece(tokenizer, vocabulary, tokens[i]);
			event.forward = forward;
			pushTrace(event);
		}
	}

	promptSeconds = timer.getElapsedTime();
	{
		T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
		if (promptSeconds > 0.0 && m_promptTokenCount > 0)
			m_promptRate = m_promptTokenCount / promptSeconds;
	}

	setState(GeneratorState::Generating, L"");

	timer.reset();
	generating = true;
	bool emitted = false;

	while (generated < maximumTokens)
	{
		if (isCancelled())
		{
			finish(GeneratorState::Cancelled, L"", L"Stopped by the user.");
			return;
		}

		float* logits = m_context->getLogits();
		if (logits == nullptr)
		{
			finish(GeneratorState::Failed, L"No logits produced.", L"Failed: the forward pass produced no logits.");
			return;
		}

		const bool interactive = isInteractive();

		int32_t token;
		SampleTrace sample;
		{
			T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
			token = m_sampler->sample(logits, m_context->getLogitCount(), m_evaluated, (trace || interactive) ? &sample : nullptr);
		}

		if (token < 0)
		{
			finish(GeneratorState::Failed, L"Sampler produced no token.", L"Failed: the sampler produced no token.");
			return;
		}

		if (interactive)
		{
			// The sampler has drawn; now the user gets to overrule it. Show
			// the distribution, wait, and carry on with whatever they picked.
			sample.interactive = true;
			sample.suggested = token;

			TraceEvent offer;
			offer.kind = TraceEvent::Kind::ChoiceOffered;
			offer.position = m_context->getPosition();
			offer.sample = sample;
			pushTrace(offer);

			setState(GeneratorState::Waiting, L"");

			int32_t chosen = -1;
			for (;;)
			{
				if (m_thread->stopped())
					return;

				if (isCancelled())
				{
					finish(GeneratorState::Cancelled, L"", L"Stopped while waiting for a choice.");
					return;
				}

				if (m_choice.wait(50))
				{
					T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
					m_choice.reset();
					chosen = m_chosenToken;
					break;
				}
			}

			if (chosen >= 0 && chosen < m_context->getLogitCount())
				token = chosen;

			sample.chosen = token;
			sample.chosenProbability = 0.0f;
			for (const auto& candidate : sample.candidates)
			{
				if (candidate.token == token)
					sample.chosenProbability = candidate.probability;
			}

			setState(GeneratorState::Generating, L"");
		}

		if (vocabulary->isEndOfGeneration(token))
		{
			// The stop token is a choice like any other, and the one most
			// worth seeing; it just has no forward pass after it.
			if (trace)
			{
				TraceEvent event;
				event.kind = TraceEvent::Kind::GeneratedToken;
				event.token = token;
				event.position = m_context->getPosition();
				event.piece = tracePiece(tokenizer, vocabulary, token);
				event.endOfGeneration = true;
				event.sample = sample;
				pushTrace(event);
			}
			break;
		}

		const std::string raw = tokenizer->decode(token);
		{
			std::string piece = raw;

			// A reply opens on a word boundary, so the space the first token
			// carries is markup rather than content.
			if (!emitted)
			{
				const size_t start = piece.find_first_not_of(" \t\n\r");
				piece = (start == std::string::npos) ? std::string() : piece.substr(start);
			}

			if (!piece.empty())
				emitted = true;

			appendText(piece);
		}

		++generated;
		{
			const double elapsed = timer.getElapsedTime();
			T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
			m_generatedTokenCount = generated;
			if (elapsed > 0.0)
				m_generateRate = generated / elapsed;
		}

		m_evaluated.push_back(token);

		ForwardTrace forward;
		const bool evaluated = m_context->evaluate(token, true, trace ? &forward : nullptr);

		if (trace)
		{
			TraceEvent event;
			event.kind = TraceEvent::Kind::GeneratedToken;
			event.token = token;
			event.position = evaluated ? forward.position : m_context->getPosition();
			event.piece = raw;
			event.sample = sample;
			event.forward = forward;
			pushTrace(event);
		}

		if (!evaluated)
		{
			finish(GeneratorState::Finished, L"Context is full; the reply was cut short.", L"Context is full; the reply was cut short.");
			return;
		}
	}

	// Anything still held back was an incomplete character that never
	// completed; emit it rather than losing it.
	{
		T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
		m_text += m_partial;
		m_partial.clear();
	}

	if (generated >= maximumTokens)
		finish(GeneratorState::Finished, L"", L"Reply reached the token limit of " + toString(maximumTokens) + L".");
	else
		finish(GeneratorState::Finished, L"", L"Reply complete; the model produced its end of turn token.");
}

bool Generator::buildPrompt(const AlignedVector< ChatMessage >& messages, int32_t maximumTokens, AlignedVector< int32_t >& outTokens, int32_t& outDroppedTurns, int32_t& outPromptBytes)
{
	const Tokenizer* tokenizer = m_model->getTokenizer();
	const ChatTemplate* chatTemplate = m_model->getChatTemplate();

	// Reserve room for the reply the caller asked for, but never more than
	// half the context; a short context would otherwise leave nothing for the
	// conversation itself.
	const int32_t contextLength = m_context->getContextLength();
	const int32_t reserve = std::max(1, std::min(maximumTokens, contextLength / c_maximumReserveShare));
	const int32_t limit = std::max(1, contextLength - reserve);

	AlignedVector< ChatMessage > kept = messages;
	outDroppedTurns = 0;

	for (;;)
	{
		const std::string prompt = chatTemplate->format(kept);

		outTokens.resize(0);
		tokenizer->encode(prompt, true, true, outTokens);

		if ((int32_t)outTokens.size() <= limit)
		{
			outPromptBytes = (int32_t)prompt.size();
			return true;
		}

		// Drop the oldest turn that is not the system prompt, and try again.
		size_t drop = kept.size();
		for (size_t i = 0; i < kept.size(); ++i)
		{
			if (kept[i].role != ChatRole::System)
			{
				drop = i;
				break;
			}
		}

		// Nothing left to drop but the turn being answered.
		if (drop >= kept.size() || kept.size() <= 1)
			return false;

		kept.erase(kept.begin() + drop);
		++outDroppedTurns;
	}
}

void Generator::appendText(const std::string& text)
{
	if (text.empty())
		return;

	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);

	m_partial += text;

	const size_t complete = getValidUtf8Length(m_partial);
	if (complete > 0)
	{
		m_text += m_partial.substr(0, complete);
		m_partial.erase(0, complete);
	}
}

void Generator::setState(GeneratorState state, const std::wstring& message)
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	m_state = state;
	if (!message.empty())
		m_message = message;
}

void Generator::pushTrace(const TraceEvent& event)
{
	T_ANONYMOUS_VAR(Acquire< Semaphore >)(m_lock);
	m_trace.push_back(event);
}

}
