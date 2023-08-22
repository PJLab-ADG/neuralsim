var dagfuncs = window.dashAgGridFunctions = window.dashAgGridFunctions || {};

dagfuncs.getDataPath = function (data) {
    return data.hierarchy;
}

dagfuncs.labelFormatter = function({ value }) {
    return `${value}%`
}

dagfuncs.barFormatter = function(params) {
    const { yValue } = params;
    return {
      fill: yValue <= 20 ? '#4fa2d9' : yValue < 60 ? '#277cb5' : '#195176',
    };
  }