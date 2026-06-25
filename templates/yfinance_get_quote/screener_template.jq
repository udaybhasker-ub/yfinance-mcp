(.results | to_entries[] | [.key, .value.data.currentPrice, .value.data.trailingPE, .value.data.recommendationKey, .value.data.sector, .value.data.longName] | @csv),
"---",
.summary.totalReturned,
.summary.totalRequested
