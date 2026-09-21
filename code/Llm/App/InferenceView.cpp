/*
 * TRAKTOR
 * Copyright (c) 2026 Anders Pistol.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "Llm/App/InferenceView.h"

#include "Core/Misc/String.h"
#include "Llm/Model.h"
#include "Llm/Tokenizer.h"
#include "Llm/Utf8.h"
#include "Llm/Vocabulary.h"
#include "Ui/Application.h"
#include "Ui/Canvas.h"
#include "Ui/Events/MouseButtonDownEvent.h"
#include "Ui/Events/MouseMoveEvent.h"
#include "Ui/Events/MouseWheelEvent.h"
#include "Ui/Events/PaintEvent.h"
#include "Ui/Events/ScrollEvent.h"
#include "Ui/Events/SizeEvent.h"
#include "Ui/ScrollBar.h"
#include "Ui/StyleSheet.h"

#include <algorithm>
#include <cmath>

namespace traktor::llm
{
namespace
{

const ui::Unit c_margin = 10_ut;
const ui::Unit c_entrySpacing = 8_ut;
const ui::Unit c_entryPadding = 8_ut;
const ui::Unit c_rowSpacing = 3_ut;
const ui::Unit c_barHeight = 8_ut;
const ui::Unit c_chipPaddingX = 5_ut;
const ui::Unit c_chipSpacing = 3_ut;
const ui::Unit c_sparklineHeight = 28_ut;
const ui::Unit c_radius = 4_ut;
const int32_t c_scrollStep = 48;

/*! Candidates listed per token; the sampler records a few more than this. */
const int32_t c_maximumCandidateRows = 8;

/*! Candidates offered when the user is choosing: everything recorded. */
const int32_t c_maximumChoiceRows = 12;

/*! Prompt tokens drawn as chips before the rest are summarized. */
const int32_t c_maximumChips = 256;

/*! Entries kept; older ones are dropped from the top of the log. */
const int32_t c_maximumEntries = 512;

/*! Share of an entry's width given to candidate names. */
const int32_t c_nameSharePercent = 38;

const wchar_t* const c_viewType = L"traktor.llm.InferenceView";

std::wstring milliseconds(double seconds)
{
	return toString(seconds * 1000.0, 1) + L" ms";
}

std::wstring percent(float probability)
{
	// A token the user forced in can sit far below anything the sampler
	// would draw; say so rather than rounding it to nothing.
	if (probability > 0.0f && probability < 0.0001f)
		return L"<0.01%";
	return toString(probability * 100.0f, (probability < 0.01f) ? 2 : 1) + L"%";
}

std::wstring rate(int32_t tokens, double seconds)
{
	if (seconds <= 0.0 || tokens <= 0)
		return L"";
	return L" (" + toString(tokens / seconds, 1) + L" tok/s)";
}

/*! Trim \a text to \a width, ending in an ellipsis if anything had to go. */
std::wstring fit(const std::wstring& text, int32_t width, const ui::FontMetric& fontMetric)
{
	if (fontMetric.getExtent(text).cx <= width)
		return text;

	const std::wstring ellipsis = L"\u2026";
	const int32_t ellipsisWidth = fontMetric.getExtent(ellipsis).cx;

	int32_t used = 0;
	size_t count = 0;
	while (count < text.size())
	{
		const wchar_t next = (count + 1 < text.size()) ? text[count + 1] : 0;
		const int32_t advance = fontMetric.getAdvance(text[count], next);
		if (used + advance + ellipsisWidth > width)
			break;
		used += advance;
		++count;
	}

	return text.substr(0, count) + ellipsis;
}

std::wstring stageColorKey(int32_t stage)
{
	return str(L"color-stage-%d", stage);
}

}

T_IMPLEMENT_RTTI_CLASS(L"traktor.llm.TokenChooseEvent", TokenChooseEvent, ui::Event)

TokenChooseEvent::TokenChooseEvent(ui::EventSubject* sender, int32_t token)
	: ui::Event(sender)
	, m_token(token)
{
}

T_IMPLEMENT_RTTI_CLASS(L"traktor.llm.InferenceView", InferenceView, ui::Widget)

bool InferenceView::create(ui::Widget* parent)
{
	if (!ui::Widget::create(parent, ui::WsDoubleBuffer | ui::WsFocus))
		return false;

	addEventHandler< ui::PaintEvent >(this, &InferenceView::eventPaint);
	addEventHandler< ui::SizeEvent >(this, &InferenceView::eventSize);
	addEventHandler< ui::MouseWheelEvent >(this, &InferenceView::eventMouseWheel);
	addEventHandler< ui::MouseMoveEvent >(this, &InferenceView::eventMouseMove);
	addEventHandler< ui::MouseButtonDownEvent >(this, &InferenceView::eventButtonDown);

	m_scrollBar = new ui::ScrollBar();
	if (!m_scrollBar->create(this, ui::ScrollBar::WsVertical))
		return false;

	m_scrollBar->addEventHandler< ui::ScrollEvent >(this, &InferenceView::eventScroll);

	m_phase = L"idle";
	return true;
}

void InferenceView::setModel(const Model* model)
{
	m_model = model;
	clear();
}

void InferenceView::setSamplerSettings(const SamplerSettings& settings)
{
	m_samplerSettings = settings;
	update();
}

void InferenceView::addEvents(const AlignedVector< TraceEvent >& events)
{
	// Text of a token as a trace shows it: chat markup included, since seeing
	// the markup go by is part of the lesson.
	const auto pieceOf = [&](int32_t token) -> std::string {
		if (!m_model)
			return std::string();
		if (m_model->getVocabulary()->getTokenType(token) == TokenType::Control)
			return m_model->getVocabulary()->getTokenText(token);
		return m_model->getTokenizer()->decode(token);
	};

	for (const auto& event : events)
	{
		switch (event.kind)
		{
		case TraceEvent::Kind::PromptBuilt:
		{
			Entry& entry = m_entries.push_back();
			entry.kind = Entry::Kind::Prompt;
			entry.expanded = true;
			entry.templateName = event.templateName;
			entry.promptBytes = event.promptBytes;
			entry.sharedTokens = event.sharedTokens;
			entry.droppedTurns = event.droppedTurns;
			entry.totalTokens = (int32_t)event.promptTokens.size();

			const int32_t fresh = entry.totalTokens - entry.sharedTokens;
			const int32_t shown = std::min(fresh, c_maximumChips);
			for (int32_t i = 0; i < shown; ++i)
			{
				const int32_t token = event.promptTokens[entry.sharedTokens + i];
				entry.promptTokens.push_back(getTokenText(token, pieceOf(token)));
			}
			if (fresh > shown)
				entry.promptTokens.push_back(L"+" + toString(fresh - shown) + L" more");

			m_phase = L"reading prompt";
			m_replyTokens = 0;
			break;
		}

		case TraceEvent::Kind::PromptToken:
		{
			// Fold into the prompt entry it belongs to, the most recent one.
			Entry* prompt = nullptr;
			for (size_t i = m_entries.size(); i > 0 && prompt == nullptr; --i)
			{
				if (m_entries[i - 1].kind == Entry::Kind::Prompt)
					prompt = &m_entries[i - 1];
			}
			if (prompt == nullptr)
				break;

			++prompt->readTokens;
			++prompt->forwardPasses;

			for (int32_t i = 0; i < ForwardTrace::StCount; ++i)
				prompt->forward.stageSeconds[i] += event.forward.stageSeconds[i];

			prompt->forward.layerCount = event.forward.layerCount;
			prompt->forward.embeddingNorm = event.forward.embeddingNorm;
			prompt->forward.residualNorms = event.forward.residualNorms;
			prompt->forward.computedLogits |= event.forward.computedLogits;
			prompt->layoutWidth = -1;

			m_latest = prompt->forward;
			m_latestPasses = prompt->forwardPasses;
			break;
		}

		case TraceEvent::Kind::ChoiceOffered:
		{
			if (!m_entries.empty() && m_entries.back().kind == Entry::Kind::Token)
			{
				m_entries.back().expanded = false;
				m_entries.back().layoutWidth = -1;
			}

			Entry& entry = m_entries.push_back();
			entry.kind = Entry::Kind::Choice;
			entry.expanded = true;
			entry.ordinal = m_replyTokens + 1;
			entry.position = event.position;
			entry.sample = event.sample;
			entry.suggestedText = getTokenText(event.sample.suggested, pieceOf(event.sample.suggested));

			for (const auto& candidate : event.sample.candidates)
				entry.candidateTexts.push_back(getTokenText(candidate.token, pieceOf(candidate.token)));

			m_hoverEntry = -1;
			m_hoverRow = -1;
			m_followTail = true;
			m_phase = L"your turn";
			break;
		}

		case TraceEvent::Kind::GeneratedToken:
		{
			// An answered offer gives way to the token it produced.
			if (!m_entries.empty() && m_entries.back().kind == Entry::Kind::Choice)
			{
				m_entries.pop_back();
				m_hoverEntry = -1;
				m_hoverRow = -1;
			}

			// The newest token is the one being studied; the one before it
			// folds up so the new one lands in view.
			if (!m_entries.empty() && m_entries.back().kind == Entry::Kind::Token)
			{
				m_entries.back().expanded = false;
				m_entries.back().layoutWidth = -1;
			}

			Entry& entry = m_entries.push_back();
			entry.kind = Entry::Kind::Token;
			entry.expanded = true;
			entry.ordinal = ++m_replyTokens;
			entry.token = event.token;
			entry.position = event.position;
			entry.text = getTokenText(event.token, event.piece);
			entry.endOfGeneration = event.endOfGeneration;
			entry.sample = event.sample;
			entry.forward = event.forward;
			entry.forwardPasses = event.endOfGeneration ? 0 : 1;

			if (event.sample.interactive)
				entry.suggestedText = getTokenText(event.sample.suggested, pieceOf(event.sample.suggested));

			for (const auto& candidate : event.sample.candidates)
				entry.candidateTexts.push_back(getTokenText(candidate.token, pieceOf(candidate.token)));

			if (!event.endOfGeneration)
			{
				m_latest = event.forward;
				m_latestPasses = 1;
			}

			m_phase = L"generating";
			break;
		}

		case TraceEvent::Kind::Finished:
		{
			// An offer nobody answered is over; a stop ended the run.
			if (!m_entries.empty() && m_entries.back().kind == Entry::Kind::Choice)
			{
				m_entries.pop_back();
				m_hoverEntry = -1;
				m_hoverRow = -1;
			}

			Entry& entry = m_entries.push_back();
			entry.kind = Entry::Kind::Finished;
			entry.outcome = event.outcome;
			entry.summary = L"prompt " + toString(event.promptTokenCount) + L" tokens in " + toString(event.promptSeconds, 2) + L" s" + rate(event.promptTokenCount, event.promptSeconds) +
				L" \u00b7 reply " + toString(event.generatedTokenCount) + L" tokens in " + toString(event.generateSeconds, 2) + L" s" + rate(event.generatedTokenCount, event.generateSeconds);

			m_phase = L"idle";
			break;
		}
		}
	}

	if (m_entries.size() > (size_t)c_maximumEntries)
		m_entries.erase(m_entries.begin(), m_entries.begin() + (m_entries.size() - c_maximumEntries));

	layout();
	update();
}

void InferenceView::clear()
{
	m_entries.clear();
	m_latest = ForwardTrace();
	m_latestPasses = 0;
	m_replyTokens = 0;
	m_phase = L"idle";
	m_contentHeight = 0;
	m_followTail = true;

	layout();
	update();
}

Ref< ui::StyleSheet > InferenceView::createStyleSheet()
{
	Ref< ui::StyleSheet > ss = new ui::StyleSheet();

	ss->setColor(c_viewType, L"background-color", Color4ub(244, 245, 248));
	ss->setColor(c_viewType, L"background-color-header", Color4ub(235, 237, 242));
	ss->setColor(c_viewType, L"background-color-entry", Color4ub(255, 255, 255));
	ss->setColor(c_viewType, L"color", Color4ub(24, 24, 28));
	ss->setColor(c_viewType, L"color-dim", Color4ub(112, 116, 126));
	ss->setColor(c_viewType, L"color-line", Color4ub(220, 222, 228));
	ss->setColor(c_viewType, L"color-accent", Color4ub(0, 122, 204));
	ss->setColor(c_viewType, L"color-on-accent", Color4ub(255, 255, 255));
	ss->setColor(c_viewType, L"background-color-kept", Color4ub(176, 208, 238));
	ss->setColor(c_viewType, L"background-color-cut", Color4ub(240, 241, 244));
	ss->setColor(c_viewType, L"background-color-chip", Color4ub(236, 237, 241));
	ss->setColor(c_viewType, L"color-chip", Color4ub(112, 116, 126));
	ss->setColor(c_viewType, L"background-color-chip-read", Color4ub(214, 232, 250));
	ss->setColor(c_viewType, L"color-chip-read", Color4ub(20, 70, 120));
	ss->setColor(c_viewType, L"background-color-hover", Color4ub(228, 240, 252));

	// One color per stage of the forward pass, in the order of ForwardTrace::Stage.
	const Color4ub stageColors[ForwardTrace::StCount] = {
		Color4ub(142, 142, 147), // embedding
		Color4ub(165, 214, 167), // attention norm
		Color4ub(66, 165, 245),	 // Q, K, V projection
		Color4ub(171, 71, 188),	 // rotary position
		Color4ub(239, 83, 80),	 // attention
		Color4ub(30, 136, 229),	 // attention output
		Color4ub(197, 225, 165), // feed forward norm
		Color4ub(255, 167, 38),	 // feed forward
		Color4ub(38, 166, 154)	 // output logits
	};
	for (int32_t i = 0; i < ForwardTrace::StCount; ++i)
		ss->setColor(c_viewType, stageColorKey(i), stageColors[i]);

	return ss;
}

int32_t InferenceView::getRowHeight() const
{
	return getFontMetric().getHeight() + pixel(c_rowSpacing);
}

ui::Rect InferenceView::getLogRect() const
{
	const ui::Rect inner = getInnerRect();
	const int32_t scrollWidth = m_scrollBar->getPreferredSize(inner.getSize()).cx;
	return ui::Rect(0, m_headerHeight, inner.getWidth() - scrollWidth, inner.getHeight());
}

std::wstring InferenceView::getTokenText(int32_t token, const std::string& piece) const
{
	std::wstring text = widenUtf8(piece);

	// Whitespace is most of what a tokenizer argues about, so make it visible.
	std::wstring shown;
	for (wchar_t ch : text)
	{
		switch (ch)
		{
		case L' ':
			shown += L'\u00b7';
			break;
		case L'\n':
			shown += L"\\n";
			break;
		case L'\r':
			shown += L"\\r";
			break;
		case L'\t':
			shown += L"\\t";
			break;
		default:
			shown += (ch < 32) ? L'?' : ch;
			break;
		}
	}

	if (shown.empty())
		shown = L"#" + toString(token);

	if (shown.size() > 24)
		shown = shown.substr(0, 23) + L"\u2026";

	return shown;
}

void InferenceView::layout()
{
	const ui::Rect inner = getInnerRect();
	const int32_t scrollWidth = m_scrollBar->getPreferredSize(inner.getSize()).cx;
	const int32_t width = inner.getWidth() - scrollWidth - 2 * pixel(c_margin);
	const int32_t rowHeight = getRowHeight();

	// Title, model, breakdown label, the bar, and the legend in two columns.
	const int32_t legendRows = (ForwardTrace::StCount + 1) / 2;
	m_headerHeight = 2 * pixel(c_margin) + 3 * rowHeight + pixel(c_barHeight) + 2 * pixel(c_rowSpacing) + legendRows * rowHeight;

	int32_t top = pixel(c_margin);
	for (auto& entry : m_entries)
	{
		// Only entries that changed, and every entry on a resize, are
		// measured again; the rest just move.
		if (entry.layoutWidth != width)
			layoutEntry(entry, width);

		entry.top = top;
		top += entry.height + pixel(c_entrySpacing);
	}

	m_contentHeight = m_entries.empty() ? 0 : top - pixel(c_entrySpacing) + pixel(c_margin);

	updateScrollBar();

	if (m_followTail)
		scrollToTail();
}

void InferenceView::layoutEntry(Entry& entry, int32_t width)
{
	const ui::FontMetric fontMetric = getFontMetric();
	const int32_t rowHeight = getRowHeight();
	const int32_t padding = pixel(c_entryPadding);
	const int32_t innerWidth = std::max(pixel(32_ut), width - 2 * padding);
	const int32_t rowSpacing = pixel(c_rowSpacing);
	const int32_t barHeight = pixel(c_barHeight);

	int32_t height = padding;

	switch (entry.kind)
	{
	case Entry::Kind::Prompt:
	{
		height += rowHeight; // Title and template.
		height += rowHeight; // Counts.

		entry.chipRects.resize(0);
		entry.chipsBottom = height;

		if (entry.expanded)
		{
			// Chips flow left to right and wrap, like the text they came from.
			const int32_t chipPaddingX = pixel(c_chipPaddingX);
			const int32_t chipSpacing = pixel(c_chipSpacing);
			int32_t x = padding;
			int32_t y = height + chipSpacing;

			for (const auto& text : entry.promptTokens)
			{
				const int32_t chipWidth = std::min(innerWidth, fontMetric.getExtent(text).cx + 2 * chipPaddingX);
				if (x + chipWidth > padding + innerWidth && x > padding)
				{
					x = padding;
					y += rowHeight + chipSpacing;
				}

				entry.chipRects.push_back(ui::Rect(ui::Point(x, y), ui::Size(chipWidth, rowHeight)));
				x += chipWidth + chipSpacing;
			}

			height = entry.promptTokens.empty() ? height + chipSpacing : y + rowHeight + chipSpacing;
			entry.chipsBottom = height;

			height += rowSpacing + rowHeight + barHeight + rowSpacing; // Forward passes and their bar.
		}
		break;
	}

	case Entry::Kind::Choice:
		height += rowHeight; // Ordinal and the invitation.
		height += rowHeight; // The sampler's suggestion and how to proceed.
		height += std::min(c_maximumChoiceRows, (int32_t)entry.sample.candidates.size()) * rowHeight;
		break;

	case Entry::Kind::Token:
	{
		height += rowHeight; // Ordinal, token and position.

		if (entry.expanded)
		{
			height += rowHeight; // Distribution label.
			height += std::min(c_maximumCandidateRows, (int32_t)entry.sample.candidates.size()) * rowHeight;
			height += rowHeight; // Sampler settings.

			if (entry.endOfGeneration)
				height += rowHeight; // Note that nothing follows.
			else
			{
				height += rowSpacing + rowHeight + barHeight + rowSpacing;			// Forward pass and its bar.
				height += rowHeight + pixel(c_sparklineHeight) + rowSpacing; // Residual norms.
			}
		}
		break;
	}

	case Entry::Kind::Finished:
		height += 2 * rowHeight;
		break;
	}

	height += padding;

	entry.height = height;
	entry.layoutWidth = width;
}

void InferenceView::updateScrollBar()
{
	// See ChatView::updateScrollBar: whole content and visible slice, both in
	// pixels, page before range.
	const ui::Rect logRect = getLogRect();
	m_scrollBar->setPage(std::max(1, logRect.getHeight()));
	m_scrollBar->setRange(std::max(0, m_contentHeight));
	m_scrollBar->update();
}

void InferenceView::scrollToTail()
{
	const ui::Rect logRect = getLogRect();
	m_scrollBar->setPosition(std::max(0, m_contentHeight - logRect.getHeight()));
	m_scrollBar->update();
}

void InferenceView::paintHeader(ui::Canvas& canvas, const ui::Rect& rc)
{
	const ui::StyleSheet* ss = getStyleSheet();
	const ui::FontMetric fontMetric = getFontMetric();
	const int32_t rowHeight = getRowHeight();
	const int32_t margin = pixel(c_margin);

	canvas.setBackground(ss->getColor(this, L"background-color-header"));
	canvas.fillRect(rc);

	canvas.setForeground(ss->getColor(this, L"color-line"));
	canvas.drawLine(rc.left, rc.bottom - 1, rc.right, rc.bottom - 1);

	const int32_t x0 = rc.left + margin;
	const int32_t x1 = rc.right - margin;
	int32_t y = rc.top + margin;

	ui::Font bold = getFont();
	bold.setBold(true);

	canvas.setFont(bold);
	canvas.setForeground(ss->getColor(this, L"color"));
	canvas.drawText(ui::Point(x0, y), L"Inference, step by step");
	canvas.setFont(getFont());

	const bool busy = (m_phase != L"idle");
	canvas.setForeground(ss->getColor(this, busy ? L"color-accent" : L"color-dim"));
	canvas.drawText(ui::Rect(x0, y, x1, y + rowHeight), m_phase, ui::AnRight, ui::AnTop);
	y += rowHeight;

	std::wstring shape = L"No model loaded.";
	if (m_model)
	{
		const ModelParameters& p = m_model->getParameters();
		shape = str(L"%d layers \u00b7 width %d \u00b7 %d heads, %d for keys and values \u00b7 vocabulary %d", p.layerCount, p.embeddingLength, p.headCount, p.headCountKv, p.vocabularyCount);
	}
	canvas.setForeground(ss->getColor(this, L"color-dim"));
	canvas.drawText(ui::Point(x0, y), fit(shape, x1 - x0, fontMetric));
	y += rowHeight;

	const double total = m_latest.getTotalSeconds();
	const std::wstring label = (m_latestPasses > 1) ? L"Where the prompt's " + toString(m_latestPasses) + L" forward passes spent their time" : L"Where the last forward pass spent its time";
	canvas.setForeground(ss->getColor(this, L"color"));
	canvas.drawText(ui::Point(x0, y), fit(label, x1 - x0 - fontMetric.getExtent(L"00000.0 ms").cx, fontMetric));
	if (total > 0.0)
	{
		canvas.setForeground(ss->getColor(this, L"color-dim"));
		canvas.drawText(ui::Rect(x0, y, x1, y + rowHeight), milliseconds(total), ui::AnRight, ui::AnTop);
	}
	y += rowHeight;

	paintStageBar(canvas, m_latest, ui::Rect(x0, y, x1, y + pixel(c_barHeight)));
	y += pixel(c_barHeight) + 2 * pixel(c_rowSpacing);

	// Legend in two columns, filled top to bottom, with each stage's share.
	const int32_t legendRows = (ForwardTrace::StCount + 1) / 2;
	const int32_t columnWidth = (x1 - x0) / 2;
	const int32_t square = fontMetric.getHeight() * 2 / 3;
	const int32_t gap = pixel(4_ut);

	for (int32_t i = 0; i < ForwardTrace::StCount; ++i)
	{
		const int32_t column = i / legendRows;
		const int32_t row = i % legendRows;
		const int32_t lx = x0 + column * columnWidth;
		const int32_t ly = y + row * rowHeight;
		const int32_t sy = ly + (rowHeight - square) / 2;

		canvas.setBackground(ss->getColor(this, stageColorKey(i)));
		canvas.fillRoundRect(ui::Rect(lx, sy, lx + square, sy + square), pixel(2_ut));

		const std::wstring share = (total > 0.0) ? toString(m_latest.stageSeconds[i] / total * 100.0, 0) + L"%" : std::wstring();
		const int32_t shareWidth = fontMetric.getExtent(L"100%").cx;

		canvas.setForeground(ss->getColor(this, (total > 0.0) ? L"color" : L"color-dim"));
		canvas.drawText(ui::Point(lx + square + gap, ly), fit(ForwardTrace::getStageName(i), columnWidth - square - 2 * gap - shareWidth - gap, fontMetric));

		if (!share.empty())
		{
			canvas.setForeground(ss->getColor(this, L"color-dim"));
			canvas.drawText(ui::Rect(lx, ly, lx + columnWidth - gap, ly + rowHeight), share, ui::AnRight, ui::AnTop);
		}
	}
}

void InferenceView::paintEntry(ui::Canvas& canvas, const Entry& entry, const ui::Rect& rc)
{
	const ui::StyleSheet* ss = getStyleSheet();

	canvas.setBackground(ss->getColor(this, L"background-color-entry"));
	canvas.fillRoundRect(rc, pixel(c_radius));

	canvas.setForeground(ss->getColor(this, L"color-line"));
	canvas.drawRoundRect(rc, pixel(c_radius));

	switch (entry.kind)
	{
	case Entry::Kind::Prompt:
		paintPrompt(canvas, entry, rc);
		break;
	case Entry::Kind::Choice:
		paintChoice(canvas, entry, (int32_t)(&entry - m_entries.c_ptr()), rc);
		break;
	case Entry::Kind::Token:
		paintToken(canvas, entry, rc);
		break;
	case Entry::Kind::Finished:
		paintFinished(canvas, entry, rc);
		break;
	}
}

void InferenceView::paintPrompt(ui::Canvas& canvas, const Entry& entry, const ui::Rect& rc)
{
	const ui::StyleSheet* ss = getStyleSheet();
	const ui::FontMetric fontMetric = getFontMetric();
	const int32_t rowHeight = getRowHeight();
	const int32_t padding = pixel(c_entryPadding);
	const int32_t x0 = rc.left + padding;
	const int32_t x1 = rc.right - padding;
	int32_t y = rc.top + padding;

	ui::Font bold = getFont();
	bold.setBold(true);

	canvas.setFont(bold);
	canvas.setForeground(ss->getColor(this, L"color"));
	canvas.drawText(ui::Point(x0, y), L"Prompt");
	canvas.setFont(getFont());

	const int32_t titleWidth = fontMetric.getExtent(L"Prompt").cx + pixel(8_ut);
	const std::wstring how = entry.templateName + L" template \u00b7 " + toString(entry.promptBytes) + L" bytes \u2192 " + toString(entry.totalTokens) + L" tokens";
	canvas.setForeground(ss->getColor(this, L"color-dim"));
	canvas.drawText(ui::Rect(x0 + titleWidth, y, x1, y + rowHeight), fit(how, x1 - x0 - titleWidth, fontMetric), ui::AnRight, ui::AnTop);
	y += rowHeight;

	const int32_t fresh = entry.totalTokens - entry.sharedTokens;
	std::wstring counts;
	if (entry.sharedTokens > 0)
		counts += toString(entry.sharedTokens) + L" already in the cache \u00b7 ";
	if (entry.readTokens < fresh)
		counts += L"reading " + toString(entry.readTokens) + L" of " + toString(fresh) + L" new";
	else
		counts += toString(fresh) + L" new read into the cache";
	if (entry.droppedTurns > 0)
		counts += L" \u00b7 " + toString(entry.droppedTurns) + L" old turn(s) dropped to fit";

	canvas.drawText(ui::Point(x0, y), fit(counts, x1 - x0, fontMetric));
	y += rowHeight;

	if (!entry.expanded)
		return;

	// Chips fill in as the tokens are read; the rest wait as outlines.
	for (size_t i = 0; i < entry.chipRects.size(); ++i)
	{
		const bool read = ((int32_t)i < entry.readTokens);
		ui::Rect chip = entry.chipRects[i];
		chip.left += rc.left;
		chip.right += rc.left;
		chip.top += rc.top;
		chip.bottom += rc.top;

		paintChip(canvas, chip, entry.promptTokens[i], L"background-color-chip-read", read ? L"color-chip-read" : L"color-dim", !read);
	}

	y = rc.top + entry.chipsBottom + pixel(c_rowSpacing);

	const double total = entry.forward.getTotalSeconds();
	const std::wstring passes = toString(entry.forwardPasses) + ((entry.forwardPasses == 1) ? L" forward pass" : L" forward passes") + L" \u00b7 logits only for the last";
	canvas.setForeground(ss->getColor(this, L"color"));
	canvas.drawText(ui::Point(x0, y), fit(passes, x1 - x0 - fontMetric.getExtent(L"00000.0 ms").cx, fontMetric));
	if (total > 0.0)
	{
		canvas.setForeground(ss->getColor(this, L"color-dim"));
		canvas.drawText(ui::Rect(x0, y, x1, y + rowHeight), milliseconds(total), ui::AnRight, ui::AnTop);
	}
	y += rowHeight;

	paintStageBar(canvas, entry.forward, ui::Rect(x0, y, x1, y + pixel(c_barHeight)));
}

void InferenceView::paintToken(ui::Canvas& canvas, const Entry& entry, const ui::Rect& rc)
{
	const ui::StyleSheet* ss = getStyleSheet();
	const ui::FontMetric fontMetric = getFontMetric();
	const int32_t rowHeight = getRowHeight();
	const int32_t padding = pixel(c_entryPadding);
	const int32_t rowSpacing = pixel(c_rowSpacing);
	const int32_t x0 = rc.left + padding;
	const int32_t x1 = rc.right - padding;
	int32_t y = rc.top + padding;

	const SampleTrace& sample = entry.sample;
	const double total = entry.forward.getTotalSeconds();

	ui::Font bold = getFont();
	bold.setBold(true);

	const std::wstring ordinal = L"#" + toString(entry.ordinal);
	canvas.setFont(bold);
	canvas.setForeground(ss->getColor(this, L"color"));
	canvas.drawText(ui::Point(x0, y), ordinal);
	canvas.setFont(getFont());

	// The token itself, as a chip in the accent color.
	const int32_t chipX = x0 + fontMetric.getExtent(ordinal).cx + pixel(8_ut);
	const int32_t chipWidth = std::min((x1 - chipX) / 2, fontMetric.getExtent(entry.text).cx + 2 * pixel(c_chipPaddingX));
	paintChip(canvas, ui::Rect(chipX, y, chipX + chipWidth, y + rowHeight), entry.text, L"color-accent", L"color-on-accent", false);

	std::wstring where = L"token " + toString(entry.token) + L" \u00b7 position " + toString(entry.position);
	if (!entry.expanded)
		where = L"p " + percent(sample.chosenProbability) + L" \u00b7 " + where;
	if (total > 0.0)
		where += L" \u00b7 " + milliseconds(total);

	const int32_t whereX = chipX + chipWidth + pixel(8_ut);
	canvas.setForeground(ss->getColor(this, L"color-dim"));
	canvas.drawText(ui::Rect(whereX, y, x1, y + rowHeight), fit(where, x1 - whereX, fontMetric), ui::AnRight, ui::AnTop);
	y += rowHeight;

	if (!entry.expanded)
		return;

	canvas.setForeground(ss->getColor(this, L"color"));
	canvas.drawText(ui::Point(x0, y), L"Top of the distribution");
	canvas.setForeground(ss->getColor(this, L"color-dim"));
	canvas.drawText(ui::Rect(x0, y, x1, y + rowHeight), L"entropy " + toString(sample.entropy, 2) + L" bits", ui::AnRight, ui::AnTop);
	y += rowHeight;

	const int32_t rows = std::min(c_maximumCandidateRows, (int32_t)sample.candidates.size());
	paintCandidates(canvas, entry, rows, x0, x1, y, -1);
	y += rows * rowHeight;

	std::wstring settings;
	if (sample.interactive)
		settings = L"You chose this; the sampler drew " + entry.suggestedText;
	else if (sample.greedy)
		settings = L"Greedy: temperature 0, the most likely token is taken";
	else
	{
		settings = L"temperature " + toString(m_samplerSettings.temperature, 1) +
			L" \u00b7 top-k " + toString(m_samplerSettings.topK) + L" \u2192 " + toString(sample.keptAfterTopK) +
			L" \u00b7 top-p " + toString(m_samplerSettings.topP, 2) + L" \u2192 " + toString(sample.keptAfterTopP);
		if (sample.penalized > 0)
			settings += L" \u00b7 " + toString(sample.penalized) + L" penalized";
	}
	canvas.setForeground(ss->getColor(this, L"color-dim"));
	canvas.drawText(ui::Point(x0, y), fit(settings, x1 - x0, fontMetric));
	y += rowHeight;

	if (entry.endOfGeneration)
	{
		canvas.drawText(ui::Point(x0, y), fit(L"End of turn token: the reply stops here, and no forward pass follows.", x1 - x0, fontMetric));
		return;
	}

	y += rowSpacing;
	canvas.setForeground(ss->getColor(this, L"color"));
	canvas.drawText(ui::Point(x0, y), L"Forward pass of this token");
	if (total > 0.0)
	{
		canvas.setForeground(ss->getColor(this, L"color-dim"));
		canvas.drawText(ui::Rect(x0, y, x1, y + rowHeight), milliseconds(total), ui::AnRight, ui::AnTop);
	}
	y += rowHeight;

	paintStageBar(canvas, entry.forward, ui::Rect(x0, y, x1, y + pixel(c_barHeight)));
	y += pixel(c_barHeight) + rowSpacing;

	// The residual stream only ever has things added to it; watching its
	// length climb layer by layer is the clearest picture of that.
	const AlignedVector< float >& norms = entry.forward.residualNorms;
	canvas.setForeground(ss->getColor(this, L"color"));
	canvas.drawText(ui::Point(x0, y), L"Residual stream |x| after each layer");
	if (!norms.empty())
	{
		canvas.setForeground(ss->getColor(this, L"color-dim"));
		canvas.drawText(ui::Rect(x0, y, x1, y + rowHeight), toString(entry.forward.embeddingNorm, 1) + L" \u2192 " + toString(norms.back(), 1), ui::AnRight, ui::AnTop);
	}
	y += rowHeight;

	const int32_t sparklineHeight = pixel(c_sparklineHeight);
	if (!norms.empty())
	{
		float peak = entry.forward.embeddingNorm;
		for (float norm : norms)
			peak = std::max(peak, norm);
		if (peak <= 0.0f)
			peak = 1.0f;

		const int32_t count = (int32_t)norms.size();
		const int32_t span = x1 - x0;

		canvas.setBackground(ss->getColor(this, L"background-color-kept"));
		for (int32_t i = 0; i < count; ++i)
		{
			const int32_t left = x0 + (int32_t)((int64_t)span * i / count);
			const int32_t right = std::max(left + 1, x0 + (int32_t)((int64_t)span * (i + 1) / count) - 1);
			const int32_t height = std::max(1, (int32_t)std::lround(norms[i] / peak * sparklineHeight));
			canvas.fillRect(ui::Rect(left, y + sparklineHeight - height, right, y + sparklineHeight));
		}
	}

	canvas.setForeground(ss->getColor(this, L"color-line"));
	canvas.drawLine(x0, y + sparklineHeight, x1, y + sparklineHeight);
}

void InferenceView::paintChoice(ui::Canvas& canvas, const Entry& entry, int32_t index, const ui::Rect& rc)
{
	const ui::StyleSheet* ss = getStyleSheet();
	const ui::FontMetric fontMetric = getFontMetric();
	const int32_t rowHeight = getRowHeight();
	const int32_t padding = pixel(c_entryPadding);
	const int32_t x0 = rc.left + padding;
	const int32_t x1 = rc.right - padding;
	int32_t y = rc.top + padding;

	// An offer is the one entry that wants something, so it gets a border
	// in the accent color to say so.
	canvas.setForeground(ss->getColor(this, L"color-accent"));
	canvas.drawRoundRect(rc, pixel(c_radius));

	ui::Font bold = getFont();
	bold.setBold(true);

	const std::wstring ordinal = L"#" + toString(entry.ordinal);
	canvas.setFont(bold);
	canvas.setForeground(ss->getColor(this, L"color"));
	canvas.drawText(ui::Point(x0, y), ordinal);
	canvas.setFont(getFont());

	const int32_t titleX = x0 + fontMetric.getExtent(ordinal).cx + pixel(8_ut);
	canvas.setForeground(ss->getColor(this, L"color-accent"));
	canvas.drawText(ui::Point(titleX, y), fit(L"Your turn: pick the next token", x1 - titleX, fontMetric));

	canvas.setForeground(ss->getColor(this, L"color-dim"));
	canvas.drawText(ui::Rect(titleX, y, x1, y + rowHeight), L"position " + toString(entry.position), ui::AnRight, ui::AnTop);
	y += rowHeight;

	// What would have happened on its own, and how to make it happen.
	const std::wstring lead = L"The sampler drew ";
	canvas.drawText(ui::Point(x0, y), lead);

	const int32_t chipX = x0 + fontMetric.getExtent(lead).cx;
	const int32_t chipWidth = std::min((x1 - chipX) / 3, fontMetric.getExtent(entry.suggestedText).cx + 2 * pixel(c_chipPaddingX));
	paintChip(canvas, ui::Rect(chipX, y, chipX + chipWidth, y + rowHeight), entry.suggestedText, L"background-color-kept", L"color", false);

	const int32_t hintX = chipX + chipWidth + pixel(4_ut);
	canvas.setForeground(ss->getColor(this, L"color-dim"));
	canvas.drawText(ui::Point(hintX, y), fit(L"; click a row to continue, or Stop.", x1 - hintX, fontMetric));
	y += rowHeight;

	const int32_t rows = std::min(c_maximumChoiceRows, (int32_t)entry.sample.candidates.size());
	paintCandidates(canvas, entry, rows, x0, x1, y, (m_hoverEntry == index) ? m_hoverRow : -1);
}

void InferenceView::paintCandidates(ui::Canvas& canvas, const Entry& entry, int32_t rows, int32_t x0, int32_t x1, int32_t y, int32_t highlightRow)
{
	const ui::StyleSheet* ss = getStyleSheet();
	const ui::FontMetric fontMetric = getFontMetric();
	const int32_t rowHeight = getRowHeight();
	const SampleTrace& sample = entry.sample;

	// One bar per candidate, scaled so the most likely fills the width; the
	// label carries the real probability. Filled bars survived the cuts. In
	// an offer nothing is chosen yet, and the sampler's draw is outlined.
	const bool offer = (entry.kind == Entry::Kind::Choice);
	const int32_t nameWidth = (x1 - x0) * c_nameSharePercent / 100;
	const int32_t percentWidth = fontMetric.getExtent(L"100.0%").cx + pixel(6_ut);
	const int32_t barX = x0 + nameWidth + pixel(6_ut);
	const int32_t barWidth = std::max(pixel(10_ut), x1 - percentWidth - barX);
	const float top = (rows > 0) ? std::max(sample.candidates[0].probability, 1e-6f) : 1.0f;

	for (int32_t i = 0; i < rows; ++i)
	{
		const SampleCandidate& candidate = sample.candidates[i];
		const bool chosen = !offer && (candidate.token == sample.chosen);
		const bool suggested = offer && (candidate.token == sample.suggested);
		const std::wstring name = (i < (int32_t)entry.candidateTexts.size()) ? entry.candidateTexts[i] : std::wstring();

		if (i == highlightRow)
		{
			canvas.setBackground(ss->getColor(this, L"background-color-hover"));
			canvas.fillRect(ui::Rect(x0 - pixel(4_ut), y, x1 + pixel(4_ut), y + rowHeight));
		}

		canvas.setForeground(ss->getColor(this, (chosen || candidate.kept || i == highlightRow) ? L"color" : L"color-dim"));
		canvas.drawText(ui::Rect(x0, y, x0 + nameWidth, y + rowHeight), fit(name, nameWidth, fontMetric), ui::AnRight, ui::AnTop);

		const int32_t width = std::max(1, (int32_t)std::lround(candidate.probability / top * barWidth));
		const ui::Rect bar(barX, y + pixel(2_ut), barX + width, y + rowHeight - pixel(2_ut));

		if (chosen)
		{
			canvas.setBackground(ss->getColor(this, L"color-accent"));
			canvas.fillRect(bar);
		}
		else if (candidate.kept)
		{
			canvas.setBackground(ss->getColor(this, L"background-color-kept"));
			canvas.fillRect(bar);
		}
		else
		{
			canvas.setBackground(ss->getColor(this, L"background-color-cut"));
			canvas.fillRect(bar);
			canvas.setForeground(ss->getColor(this, L"color-line"));
			canvas.drawRect(bar);
		}

		if (suggested)
		{
			canvas.setForeground(ss->getColor(this, L"color-accent"));
			canvas.drawRect(ui::Rect(bar.left - 1, bar.top - 1, std::max(bar.right, barX + barWidth) + 1, bar.bottom + 1));
		}

		canvas.setForeground(ss->getColor(this, (chosen || suggested) ? L"color" : L"color-dim"));
		canvas.drawText(ui::Rect(x1 - percentWidth, y, x1, y + rowHeight), percent(candidate.probability), ui::AnRight, ui::AnTop);

		y += rowHeight;
	}
}

void InferenceView::paintFinished(ui::Canvas& canvas, const Entry& entry, const ui::Rect& rc)
{
	const ui::StyleSheet* ss = getStyleSheet();
	const ui::FontMetric fontMetric = getFontMetric();
	const int32_t rowHeight = getRowHeight();
	const int32_t padding = pixel(c_entryPadding);
	const int32_t x0 = rc.left + padding;
	const int32_t x1 = rc.right - padding;
	int32_t y = rc.top + padding;

	ui::Font bold = getFont();
	bold.setBold(true);

	canvas.setFont(bold);
	canvas.setForeground(ss->getColor(this, L"color"));
	canvas.drawText(ui::Point(x0, y), fit(entry.outcome, x1 - x0, fontMetric));
	canvas.setFont(getFont());
	y += rowHeight;

	canvas.setForeground(ss->getColor(this, L"color-dim"));
	canvas.drawText(ui::Point(x0, y), fit(entry.summary, x1 - x0, fontMetric));
}

void InferenceView::paintStageBar(ui::Canvas& canvas, const ForwardTrace& forward, const ui::Rect& rc)
{
	const ui::StyleSheet* ss = getStyleSheet();

	canvas.setBackground(ss->getColor(this, L"background-color-cut"));
	canvas.fillRect(rc);

	const double total = forward.getTotalSeconds();
	if (total <= 0.0)
		return;

	// Segments in stage order, each as wide as its share of the total.
	const int32_t width = rc.getWidth();
	int32_t x = rc.left;
	double accumulated = 0.0;

	for (int32_t i = 0; i < ForwardTrace::StCount; ++i)
	{
		accumulated += forward.stageSeconds[i];
		const int32_t end = rc.left + (int32_t)std::lround(accumulated / total * width);
		if (end > x)
		{
			canvas.setBackground(ss->getColor(this, stageColorKey(i)));
			canvas.fillRect(ui::Rect(x, rc.top, end, rc.bottom));
			x = end;
		}
	}
}

void InferenceView::paintChip(ui::Canvas& canvas, const ui::Rect& rc, const std::wstring& text, const wchar_t* backgroundKey, const wchar_t* colorKey, bool outline)
{
	const ui::StyleSheet* ss = getStyleSheet();
	const ui::FontMetric fontMetric = getFontMetric();
	const int32_t paddingX = pixel(c_chipPaddingX);

	if (outline)
	{
		canvas.setForeground(ss->getColor(this, L"color-line"));
		canvas.drawRoundRect(rc, pixel(c_radius));
	}
	else
	{
		canvas.setBackground(ss->getColor(this, backgroundKey));
		canvas.fillRoundRect(rc, pixel(c_radius));
	}

	canvas.setForeground(ss->getColor(this, colorKey));
	canvas.drawText(ui::Rect(rc.left + paddingX, rc.top, rc.right - paddingX, rc.bottom), fit(text, rc.getWidth() - 2 * paddingX, fontMetric), ui::AnLeft, ui::AnCenter);
}

void InferenceView::eventPaint(ui::PaintEvent* event)
{
	ui::Canvas& canvas = event->getCanvas();
	const ui::StyleSheet* ss = getStyleSheet();
	const ui::Rect inner = getInnerRect();

	canvas.setBackground(ss->getColor(this, L"background-color"));
	canvas.fillRect(inner);

	const int32_t scrollWidth = m_scrollBar->getPreferredSize(inner.getSize()).cx;
	const int32_t width = inner.getWidth() - scrollWidth;

	paintHeader(canvas, ui::Rect(0, 0, width, m_headerHeight));

	const ui::Rect logRect = getLogRect();
	const int32_t scroll = m_scrollBar->getPosition();
	const int32_t margin = pixel(c_margin);

	if (m_entries.empty())
	{
		canvas.setForeground(ss->getColor(this, L"color-dim"));
		canvas.drawText(logRect, m_model ? L"Send a message to watch the engine work." : L"Open a model to begin.", ui::AnCenter, ui::AnCenter);
	}

	canvas.setClipRect(logRect);

	for (const auto& entry : m_entries)
	{
		const int32_t top = logRect.top + entry.top - scroll;
		if (top + entry.height < logRect.top)
			continue;
		if (top > logRect.bottom)
			break;

		paintEntry(canvas, entry, ui::Rect(ui::Point(margin, top), ui::Size(width - 2 * margin, entry.height)));
	}

	canvas.resetClipRect();

	event->consume();
}

void InferenceView::eventSize(ui::SizeEvent* event)
{
	layout();

	const ui::Rect inner = getInnerRect();
	const int32_t width = m_scrollBar->getPreferredSize(inner.getSize()).cx;
	m_scrollBar->setRect(ui::Rect(ui::Point(inner.getWidth() - width, m_headerHeight), ui::Size(width, inner.getHeight() - m_headerHeight)));
}

void InferenceView::eventScroll(ui::ScrollEvent* event)
{
	// Only stay pinned to the newest entry while the view is at the end.
	const ui::Rect logRect = getLogRect();
	m_followTail = (m_scrollBar->getPosition() >= m_contentHeight - logRect.getHeight());

	update();
}

void InferenceView::eventMouseWheel(ui::MouseWheelEvent* event)
{
	const ui::Rect logRect = getLogRect();
	const int32_t range = std::max(0, m_contentHeight - logRect.getHeight());

	int32_t position = m_scrollBar->getPosition() - event->getRotation() * pixel(ui::Unit(c_scrollStep));
	position = std::max(0, std::min(position, range));

	m_scrollBar->setPosition(position);
	m_scrollBar->update();

	m_followTail = (position >= range);
	update();
}

bool InferenceView::hitChoice(const ui::Point& position, int32_t& outEntry, int32_t& outRow) const
{
	const ui::Rect logRect = getLogRect();
	if (!logRect.inside(position))
		return false;

	const int32_t rowHeight = getRowHeight();
	const int32_t y = position.y - logRect.top + m_scrollBar->getPosition();

	for (int32_t i = 0; i < (int32_t)m_entries.size(); ++i)
	{
		const Entry& entry = m_entries[i];
		if (entry.kind != Entry::Kind::Choice || y < entry.top || y >= entry.top + entry.height)
			continue;

		// Same geometry as paintChoice: padding, two rows of text, then the
		// candidates.
		const int32_t rowsTop = entry.top + pixel(c_entryPadding) + 2 * rowHeight;
		const int32_t rows = std::min(c_maximumChoiceRows, (int32_t)entry.sample.candidates.size());
		const int32_t row = (y - rowsTop) / rowHeight;

		if (y >= rowsTop && row >= 0 && row < rows)
		{
			outEntry = i;
			outRow = row;
			return true;
		}
		return false;
	}

	return false;
}

void InferenceView::eventMouseMove(ui::MouseMoveEvent* event)
{
	int32_t hoverEntry = -1;
	int32_t hoverRow = -1;
	hitChoice(event->getPosition(), hoverEntry, hoverRow);

	if (hoverEntry != m_hoverEntry || hoverRow != m_hoverRow)
	{
		m_hoverEntry = hoverEntry;
		m_hoverRow = hoverRow;
		update();
	}
}

void InferenceView::eventButtonDown(ui::MouseButtonDownEvent* event)
{
	const ui::Point position = event->getPosition();
	const ui::Rect logRect = getLogRect();
	if (!logRect.inside(position))
		return;

	// A row of an offer is an answer, not a fold.
	int32_t choiceEntry = -1;
	int32_t choiceRow = -1;
	if (hitChoice(position, choiceEntry, choiceRow))
	{
		TokenChooseEvent chooseEvent(this, m_entries[choiceEntry].sample.candidates[choiceRow].token);
		raiseEvent(&chooseEvent);
		return;
	}

	const int32_t y = position.y - logRect.top + m_scrollBar->getPosition();
	for (auto& entry : m_entries)
	{
		if (y >= entry.top && y < entry.top + entry.height)
		{
			if (entry.kind == Entry::Kind::Choice)
				return;

			entry.expanded = !entry.expanded;
			entry.layoutWidth = -1;
			break;
		}
	}

	// Clicking is reading; stop chasing the tail so the entry stays put.
	m_followTail = false;

	layout();
	update();
}

}
