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

#include <string>

namespace traktor::llm
{

/*! Where one forward pass spent its time, and what it did to the residual.
 * \ingroup Llm
 *
 * Filled in by Context::evaluate when asked. Every field is something the
 * pass computes anyway or can measure in passing; the trace is meant to
 * show the network at work, not to change what it does.
 */
class ForwardTrace
{
public:
	/*! Stages of Context::evaluate, in the order they run within a layer. */
	enum Stage
	{
		StEmbedding,	   //!< Embedding row lookup and decode.
		StAttentionNorm,   //!< RMS norm feeding attention.
		StProjection,	   //!< Query, key and value projections, with bias.
		StRotary,		   //!< Rotary position applied to query and key.
		StAttention,	   //!< Scores against the cache, softmax, weighted values.
		StAttentionOutput, //!< Output projection and residual add.
		StFeedForwardNorm, //!< RMS norm feeding the feed forward block.
		StFeedForward,	   //!< Gate, up, SiLU, down, and residual add.
		StOutput,		   //!< Final norm and the projection to logits.
		StCount
	};

	int32_t token = -1;
	int32_t position = -1;
	int32_t layerCount = 0;
	bool computedLogits = false;

	/*! Seconds spent in each stage, summed over the layers. */
	double stageSeconds[StCount] = {};

	/*! Magnitude of the residual stream as it enters the first layer. */
	float embeddingNorm = 0.0f;

	/*! Magnitude of the residual stream after each layer.
	 *
	 * The residual only ever has things added to it, so this normally
	 * climbs layer by layer; the final norm exists to undo that growth.
	 */
	AlignedVector< float > residualNorms;

	double getTotalSeconds() const
	{
		double total = 0.0;
		for (int32_t i = 0; i < StCount; ++i)
			total += stageSeconds[i];
		return total;
	}

	static const wchar_t* getStageName(int32_t stage)
	{
		switch (stage)
		{
		case StEmbedding:
			return L"embedding";
		case StAttentionNorm:
			return L"attention norm";
		case StProjection:
			return L"Q, K, V projection";
		case StRotary:
			return L"rotary position";
		case StAttention:
			return L"attention";
		case StAttentionOutput:
			return L"attention output";
		case StFeedForwardNorm:
			return L"feed forward norm";
		case StFeedForward:
			return L"feed forward";
		case StOutput:
			return L"output logits";
		default:
			return L"";
		}
	}
};

/*! One token the sampler considered. */
class SampleCandidate
{
public:
	int32_t token = -1;
	float probability = 0.0f; //!< After temperature, before any cut.
	bool kept = false;		  //!< Survived top-k and top-p, so could have been drawn.
};

/*! How the sampler arrived at its choice.
 * \ingroup Llm
 */
class SampleTrace
{
public:
	/*! Most likely candidates first. Only the tokens the sampler sorted are
	 * known in order, so this holds at most the top-k count. */
	AlignedVector< SampleCandidate > candidates;

	bool greedy = false;		 //!< Temperature was zero; the most likely token was taken.
	int32_t penalized = 0;		 //!< Recent tokens whose logit the repetition penalty lowered.
	int32_t keptAfterTopK = 0;	 //!< Candidates left after the top-k cut.
	int32_t keptAfterTopP = 0;	 //!< Candidates left after the nucleus cut; the draw was among these.
	int32_t chosen = -1;
	float chosenProbability = 0.0f;

	/*! Entropy of the whole distribution in bits: zero when the model is
	 * certain, log2(vocabulary) when it has no idea. */
	float entropy = 0.0f;

	/*! The user picked the token rather than the sampler; what the sampler
	 * drew, and would have used, is in suggested. */
	bool interactive = false;
	int32_t suggested = -1;
};

/*! One thing the generator did, for a user interface to show.
 * \ingroup Llm
 *
 * Produced in order and collected with Generator::flushTrace. A prompt
 * yields one PromptBuilt followed by a PromptToken per position read into
 * the cache; a reply yields one GeneratedToken per token drawn, including
 * the end of turn token that stops it; and every run ends with Finished.
 * In interactive mode a ChoiceOffered precedes each GeneratedToken, and the
 * generator waits on it until Generator::chooseToken answers.
 */
class TraceEvent
{
public:
	enum class Kind
	{
		PromptBuilt,
		PromptToken,
		ChoiceOffered, //!< The distribution is in sample; the next token is the user's to pick.
		GeneratedToken,
		Finished
	};

	Kind kind = Kind::Finished;

	//@name PromptBuilt
	//@{

	std::wstring templateName;
	int32_t promptBytes = 0;			  //!< Length of the formatted prompt text.
	int32_t sharedTokens = 0;			  //!< Leading tokens the cache already held.
	int32_t droppedTurns = 0;			  //!< Old turns left out to fit the context.
	AlignedVector< int32_t > promptTokens; //!< Every token of the prompt, shared ones first.

	//@}

	//@name PromptToken and GeneratedToken
	//@{

	int32_t token = -1;
	int32_t position = -1;
	std::string piece;			 //!< Token text as raw bytes, possibly a partial character.
	bool endOfGeneration = false; //!< The token stops the reply; no forward pass follows it.
	ForwardTrace forward;
	SampleTrace sample; //!< GeneratedToken only.

	//@}

	//@name Finished
	//@{

	std::wstring outcome; //!< Why the run ended, in a sentence.
	int32_t promptTokenCount = 0;
	int32_t generatedTokenCount = 0;
	double promptSeconds = 0.0;
	double generateSeconds = 0.0;

	//@}
};

}
