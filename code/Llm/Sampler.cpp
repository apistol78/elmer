/*
 * TRAKTOR
 * Copyright (c) 2026 Anders Pistol.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "Llm/Sampler.h"

#include "Core/Date/DateTime.h"
#include "Llm/Ops.h"
#include "Llm/Trace.h"

#include <algorithm>
#include <cmath>

namespace traktor::llm
{
namespace
{

/*! How many of the leading candidates a trace records. */
const int32_t c_traceCandidates = 12;

/*! Entropy in bits of \a probabilities, which must sum to one. */
float entropyBits(const float* probabilities, int32_t count)
{
	double sum = 0.0;
	for (int32_t i = 0; i < count; ++i)
	{
		const double p = probabilities[i];
		if (p > 0.0)
			sum -= p * std::log2(p);
	}
	return (float)sum;
}

}

T_IMPLEMENT_RTTI_CLASS(L"traktor.llm.Sampler", Sampler, Object)

Sampler::Sampler(const SamplerSettings& settings)
{
	setSettings(settings);
}

void Sampler::setSettings(const SamplerSettings& settings)
{
	const bool reseed = !m_seeded || m_settings.seed != settings.seed;

	m_settings = settings;

	if (reseed)
	{
		uint32_t seed = settings.seed;
		if (seed == 0)
			seed = (uint32_t)DateTime::now().getSecondsSinceEpoch() ^ 0x9e3779b9;

		m_random = Random(seed);
		m_seeded = true;
	}
}

int32_t Sampler::sample(float* logits, int32_t count, const AlignedVector< int32_t >& history, SampleTrace* outTrace)
{
	if (count <= 0)
		return -1;

	if (outTrace)
		*outTrace = SampleTrace();

	// Discourage whatever was said recently. Positive logits are divided and
	// negative ones multiplied, so the penalty always moves a token down.
	if (m_settings.repeatPenalty > 1.0f && m_settings.repeatWindow > 0)
	{
		const int32_t from = std::max(0, (int32_t)history.size() - m_settings.repeatWindow);
		for (int32_t i = from; i < (int32_t)history.size(); ++i)
		{
			const int32_t token = history[i];
			if (token < 0 || token >= count)
				continue;

			if (logits[token] > 0.0f)
				logits[token] /= m_settings.repeatPenalty;
			else
				logits[token] *= m_settings.repeatPenalty;

			if (outTrace)
				++outTrace->penalized;
		}
	}

	if (m_settings.temperature <= 0.0f)
	{
		int32_t best = 0;
		for (int32_t i = 1; i < count; ++i)
		{
			if (logits[i] > logits[best])
				best = i;
		}

		// Greedy needs no probabilities, but a trace is about showing the
		// distribution the choice was made from, so build it anyway.
		if (outTrace)
		{
			softmax(logits, (uint32_t)count);

			m_candidates.resize(count);
			for (int32_t i = 0; i < count; ++i)
				m_candidates[i] = { i, logits[i] };

			const int32_t shown = std::min(c_traceCandidates, count);
			std::partial_sort(m_candidates.begin(), m_candidates.begin() + shown, m_candidates.end(), [](const Candidate& a, const Candidate& b) {
				return a.probability > b.probability;
			});

			outTrace->greedy = true;
			outTrace->keptAfterTopK = 1;
			outTrace->keptAfterTopP = 1;
			outTrace->chosen = best;
			outTrace->chosenProbability = logits[best];
			outTrace->entropy = entropyBits(logits, count);

			for (int32_t i = 0; i < shown; ++i)
			{
				SampleCandidate& candidate = outTrace->candidates.push_back();
				candidate.token = m_candidates[i].token;
				candidate.probability = m_candidates[i].probability;
				candidate.kept = (candidate.token == best);
			}
		}

		return best;
	}

	const float scale = 1.0f / m_settings.temperature;
	for (int32_t i = 0; i < count; ++i)
		logits[i] *= scale;

	softmax(logits, (uint32_t)count);

	if (outTrace)
		outTrace->entropy = entropyBits(logits, count);

	m_candidates.resize(count);
	for (int32_t i = 0; i < count; ++i)
		m_candidates[i] = { i, logits[i] };

	int32_t keep = count;

	// Top k first; it bounds the sort that nucleus sampling needs.
	if (m_settings.topK > 0 && m_settings.topK < keep)
	{
		std::partial_sort(m_candidates.begin(), m_candidates.begin() + m_settings.topK, m_candidates.begin() + keep, [](const Candidate& a, const Candidate& b) {
			return a.probability > b.probability;
		});
		keep = m_settings.topK;
	}
	else
	{
		std::sort(m_candidates.begin(), m_candidates.begin() + keep, [](const Candidate& a, const Candidate& b) {
			return a.probability > b.probability;
		});
	}

	// Only this many are known in order; anything past the top-k boundary
	// was never sorted.
	const int32_t sorted = keep;

	if (outTrace)
		outTrace->keptAfterTopK = keep;

	// Nucleus: keep the shortest prefix holding at least topP of the mass.
	if (m_settings.topP < 1.0f)
	{
		float cumulative = 0.0f;
		for (int32_t i = 0; i < keep; ++i)
		{
			cumulative += m_candidates[i].probability;
			if (cumulative >= m_settings.topP)
			{
				keep = i + 1;
				break;
			}
		}
	}

	if (outTrace)
	{
		outTrace->keptAfterTopP = keep;

		const int32_t shown = std::min(c_traceCandidates, sorted);
		for (int32_t i = 0; i < shown; ++i)
		{
			SampleCandidate& candidate = outTrace->candidates.push_back();
			candidate.token = m_candidates[i].token;
			candidate.probability = m_candidates[i].probability;
			candidate.kept = (i < keep);
		}
	}

	float total = 0.0f;
	for (int32_t i = 0; i < keep; ++i)
		total += m_candidates[i].probability;

	int32_t chosen = keep - 1;
	if (total <= 0.0f)
		chosen = 0;
	else
	{
		float target = m_random.nextFloat() * total;
		for (int32_t i = 0; i < keep; ++i)
		{
			target -= m_candidates[i].probability;
			if (target <= 0.0f)
			{
				chosen = i;
				break;
			}
		}
	}

	if (outTrace)
	{
		outTrace->chosen = m_candidates[chosen].token;
		outTrace->chosenProbability = m_candidates[chosen].probability;
	}

	return m_candidates[chosen].token;
}

}
