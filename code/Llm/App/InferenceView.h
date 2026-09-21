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
#include "Core/Ref.h"
#include "Llm/Sampler.h"
#include "Llm/Trace.h"
#include "Ui/Event.h"
#include "Ui/Rect.h"
#include "Ui/Widget.h"

#include <string>

namespace traktor::ui
{

class Canvas;
class FontMetric;
class ScrollBar;
class StyleSheet;

}

namespace traktor::llm
{

class Model;

/*! The user picked a token from an offered distribution.
 * \ingroup Llm
 */
class TokenChooseEvent : public ui::Event
{
	T_RTTI_CLASS;

public:
	explicit TokenChooseEvent(ui::EventSubject* sender, int32_t token);

	int32_t getToken() const { return m_token; }

private:
	int32_t m_token;
};

/*! Step by step account of what the engine is doing, beside the transcript.
 * \ingroup Llm
 *
 * Shows the reply being made rather than the reply: the prompt becoming
 * tokens and being read into the cache, then for every token drawn the
 * distribution it came from, the cuts the sampler made, where the forward
 * pass spent its time and how the residual stream grew through the layers.
 * A fixed header keeps the latest time breakdown in view; the log below
 * scrolls, and an entry expands or collapses when clicked.
 *
 * In interactive mode the newest entry is an offer: the distribution with
 * clickable rows. Clicking one raises a TokenChooseEvent.
 */
class InferenceView : public ui::Widget
{
	T_RTTI_CLASS;

public:
	bool create(ui::Widget* parent);

	/*! Model whose vocabulary names the tokens shown; null clears it. */
	void setModel(const Model* model);

	/*! Settings shown against each sampling decision. */
	void setSamplerSettings(const SamplerSettings& settings);

	/*! Show \a events, in the order the generator produced them. */
	void addEvents(const AlignedVector< TraceEvent >& events);

	void clear();

	/*! Colors used by the view, merged over the default sheet. */
	static Ref< ui::StyleSheet > createStyleSheet();

private:
	/*! One block of the log. A prompt is a single entry that fills in as its
	 * tokens are read; every generated token is an entry of its own. */
	class Entry
	{
	public:
		enum class Kind
		{
			Prompt,
			Choice, //!< A distribution waiting for the user to pick from it.
			Token,
			Finished
		};

		Kind kind = Kind::Prompt;
		bool expanded = true;

		//@name Prompt
		//@{

		std::wstring templateName;
		int32_t promptBytes = 0;
		int32_t sharedTokens = 0;
		int32_t droppedTurns = 0;
		int32_t totalTokens = 0;					 //!< Cached and new together.
		int32_t readTokens = 0;						 //!< New tokens read so far.
		AlignedVector< std::wstring > promptTokens; //!< Display text of the tokens not already cached.

		//@}

		//@name Token
		//@{

		int32_t ordinal = 0; //!< Index within the reply, from one.
		int32_t token = -1;
		int32_t position = -1;
		std::wstring text;
		bool endOfGeneration = false;
		SampleTrace sample;
		AlignedVector< std::wstring > candidateTexts;
		std::wstring suggestedText; //!< The sampler's own draw, when the user chose instead.

		//@}

		//@name Finished
		//@{

		std::wstring outcome;
		std::wstring summary;

		//@}

		/*! Forward pass timing: one pass for a token, the sum for a prompt. */
		ForwardTrace forward;
		int32_t forwardPasses = 0;

		//@name Layout, valid for the width it was computed at.
		//@{

		int32_t top = 0;
		int32_t height = 0;
		int32_t layoutWidth = -1;
		AlignedVector< ui::Rect > chipRects; //!< Prompt token chips, relative to the entry.
		int32_t chipsBottom = 0;			 //!< Where the chips end, relative to the entry.

		//@}
	};

	Ref< ui::ScrollBar > m_scrollBar;
	Ref< const Model > m_model;
	SamplerSettings m_samplerSettings;
	AlignedVector< Entry > m_entries;
	ForwardTrace m_latest; //!< Breakdown shown in the header.
	int32_t m_latestPasses = 0;
	std::wstring m_phase;
	int32_t m_replyTokens = 0;
	int32_t m_headerHeight = 0;
	int32_t m_contentHeight = 0;
	bool m_followTail = true;
	int32_t m_hoverEntry = -1; //!< Choice entry under the pointer, or -1.
	int32_t m_hoverRow = -1;   //!< Candidate row under the pointer within it.

	int32_t getRowHeight() const;

	/*! Candidate row of a choice entry at \a position, if any. */
	bool hitChoice(const ui::Point& position, int32_t& outEntry, int32_t& outRow) const;

	/*! Log area: everything under the header, less the scroll bar. */
	ui::Rect getLogRect() const;

	std::wstring getTokenText(int32_t token, const std::string& piece) const;

	void layout();

	void layoutEntry(Entry& entry, int32_t width);

	void updateScrollBar();

	void scrollToTail();

	void paintHeader(ui::Canvas& canvas, const ui::Rect& rc);

	void paintEntry(ui::Canvas& canvas, const Entry& entry, const ui::Rect& rc);

	void paintPrompt(ui::Canvas& canvas, const Entry& entry, const ui::Rect& rc);

	void paintToken(ui::Canvas& canvas, const Entry& entry, const ui::Rect& rc);

	void paintChoice(ui::Canvas& canvas, const Entry& entry, int32_t index, const ui::Rect& rc);

	void paintFinished(ui::Canvas& canvas, const Entry& entry, const ui::Rect& rc);

	/*! One row per candidate: name, bar and probability, starting at \a y. */
	void paintCandidates(ui::Canvas& canvas, const Entry& entry, int32_t rows, int32_t x0, int32_t x1, int32_t y, int32_t highlightRow);

	/*! Stacked bar of stage times; returns nothing drawn when the total is zero. */
	void paintStageBar(ui::Canvas& canvas, const ForwardTrace& forward, const ui::Rect& rc);

	void paintChip(ui::Canvas& canvas, const ui::Rect& rc, const std::wstring& text, const wchar_t* backgroundKey, const wchar_t* colorKey, bool outline);

	void eventPaint(ui::PaintEvent* event);

	void eventSize(ui::SizeEvent* event);

	void eventScroll(ui::ScrollEvent* event);

	void eventMouseWheel(ui::MouseWheelEvent* event);

	void eventMouseMove(ui::MouseMoveEvent* event);

	void eventButtonDown(ui::MouseButtonDownEvent* event);
};

}
